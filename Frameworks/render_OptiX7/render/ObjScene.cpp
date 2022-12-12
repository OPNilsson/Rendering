#include <vector>
#include <string>
#include <map>

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_stubs.h>

#include <sutil/Exception.h>
#include <sutil/Matrix.h>
#include <sutil/Scene.h>
#include <sutil/Record.h>
#include <sutil/sutil.h>

#include <GLFW/glfw3.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <support/tinyobjloader/tiny_obj_loader.h>

#include "Directional.h"
#include "HDRLoader.h"
#include "ObjScene.h"

using namespace sutil;
using namespace std;

LaunchParams launch_params;
LaunchParams* d_params = nullptr;

namespace
{
  unsigned int MAX_DEPTH = 10u;
  unsigned int TRANSLUCENT_SAMPLES = 1000u;
  
  typedef Record<HitGroupData> HitGroupRecord;

  vector<tinyobj::shape_t> obj_shapes;
  vector<tinyobj::material_t> obj_materials;
  int32_t bufidx = 0;
}

ObjScene::~ObjScene()
{
  try
  {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(launch_params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(launch_params.lights.data)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
  }
  catch(exception& e)
  {
    cerr << "Caught exception: " << e.what() << "\n";
  }
}

void ObjScene::initScene()
{
  scene.cleanup();
  if(use_envmap)
  {
    bool is_hdr = envfile.compare(envfile.length() - 3, 3, "hdr") == 0;
    if(is_hdr)
    {
      HDRLoader hdr(envfile);
      if(hdr.failed()) 
      {
        cerr << "Could not load HDR environment map called: " << envfile << endl;
        use_envmap = false;
      }
      scene.addImage(hdr.width(), hdr.height(), 32, 4, hdr.raster());
    }
    else
    {
      ImageBuffer img = loadImage(envfile.c_str());
      if(img.pixel_format != UNSIGNED_BYTE4)
      {
        cerr << "Environment map texture image with unknown pixel format: " << envfile << endl;
        use_envmap = false;
      }
      scene.addImage(img.width, img.height, 8, 4, img.data);
    }
    if(use_envmap)
      scene.addSampler(cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, 0, is_hdr);
  }
  if(!filenames.empty())
  {
    //loadScene(filenames[0], scene);
    loadObjs();
    scene.createContext();
    scene.buildMeshAccels();
    scene.buildInstanceAccel();

    OPTIX_CHECK(optixInit()); // Need to initialize function table
    createPTXModule();
    createProgramGroups();
    createPipeline();
    createSBT();
    
    bbox.invalidate();
    for(const auto mesh : scene.meshes())
      if(mesh->world_aabb.area() < 1.0e6f) // Objects with a very large bounding box are considered background
        bbox.include(mesh->world_aabb);
    cout << "Scene bounding box maximum extent: " << bbox.maxExtent() << endl;
    
    initCameraState();
    initLaunchParams(scene);
  }
}

void ObjScene::initLaunchParams(const Scene& scene)
{
  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void**>(&launch_params.accum_buffer),
    width*height*sizeof(float4)
  ));
  launch_params.frame_buffer = nullptr; // Will be set when output buffer is mapped
  launch_params.subframe_index = 0u;
  launch_params.max_depth = MAX_DEPTH;

  const float loffset = bbox.maxExtent();

  // Add light sources depending on chosen shader
  if(shadername == "arealight")
  {
    if(!extract_area_lights())
    {
      cerr << "Error: no area lights in scene. "
           << "You cannot use the area light shader if there are no emissive objects in the scene. "
           << "Objects are emissive if their ambient color is not zero."
           << endl;
      exit(0);
    }
  }
  else
    add_default_light();

  if(use_envmap)
    launch_params.envmap = scene.getSampler(0);

  //CUDA_CHECK( cudaStreamCreate( &stream ) );
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(LaunchParams)));

  launch_params.handle = scene.traversableHandle();
}

void ObjScene::initCameraState()
{
  camera.setFovY(45.0f);
  camera.setLookat(bbox.center());
  camera.setEye(bbox.center() + make_float3(0.0f, 0.0f, 1.8f*bbox.maxExtent()));
}

void ObjScene::handleCameraUpdate()
{
  camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
  launch_params.eye = camera.eye();
  camera.UVWFrame(launch_params.U, launch_params.V, launch_params.W);
  /*
  cerr
      << "Updating camera:\n"
      << "\tU: " << launch_params.U.x << ", " << launch_params.U.y << ", " << launch_params.U.z << endl
      << "\tV: " << launch_params.V.x << ", " << launch_params.V.y << ", " << launch_params.V.z << endl
      << "\tW: " << launch_params.W.x << ", " << launch_params.W.y << ", " << launch_params.W.z << endl;
      */
}

void ObjScene::handleResize(CUDAOutputBuffer<uchar4>& output_buffer, int32_t w, int32_t h)
{
  width = w;
  height = h;
  output_buffer.resize(width, height);

  // Realloc accumulation buffer
  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(launch_params.accum_buffer)));
  CUDA_CHECK(cudaMalloc(
    reinterpret_cast<void**>(&launch_params.accum_buffer),
    width*height*sizeof(float4)
  ));
}

void ObjScene::launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer)
{
  // Launch
  uchar4* result_buffer_data = output_buffer.map();
  launch_params.frame_buffer = result_buffer_data;
  CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
    &launch_params,
    sizeof(LaunchParams),
    cudaMemcpyHostToDevice,
    0 // stream
  ));

  OPTIX_CHECK(optixLaunch(
    m_pipeline,
    0,             // stream
    reinterpret_cast<CUdeviceptr>(d_params),
    sizeof(LaunchParams),
    &m_sbt,
    width,  // launch width
    height, // launch height
    1       // launch depth
  ));
  output_buffer.unmap();
  CUDA_SYNC_CHECK();
}

void ObjScene::createPTXModule()
{
  OptixModuleCompileOptions module_compile_options = {};
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

  m_pipeline_compile_options = {};
  m_pipeline_compile_options.usesMotionBlur = false;
  m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  m_pipeline_compile_options.numPayloadValues = NUM_PAYLOAD_VALUES;
  m_pipeline_compile_options.numAttributeValues = 2; // TODO
  m_pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE; // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
  m_pipeline_compile_options.pipelineLaunchParamsVariableName = "launch_params";

  size_t inputSize = 0;
  const string ptx(getInputData(nullptr, "render", "shaders.cu", inputSize));

  m_ptx_module = {};
  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
    scene.context(),
    &module_compile_options,
    &m_pipeline_compile_options,
    ptx.c_str(),
    ptx.size(),
    log,
    &sizeof_log,
    &m_ptx_module
  ));
}

void ObjScene::createProgramGroups()
{
  OptixProgramGroupOptions program_group_options = {};
  char log[2048];
  size_t sizeof_log = sizeof(log);

  //
  // Ray generation
  //
  {

    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = m_ptx_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
      scene.context(),
      &raygen_prog_group_desc,
      1,                             // num program groups
      &program_group_options,
      log,
      &sizeof_log,
      &m_raygen_prog_group
    ));
  }

  //
  // Miss
  //
  {
    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = m_ptx_module;
    miss_prog_group_desc.miss.entryFunctionName = use_envmap ? "__miss__envmap_radiance" : "__miss__constant_radiance";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
      scene.context(),
      &miss_prog_group_desc,
      1,                             // num program groups
      &program_group_options,
      log,
      &sizeof_log,
      &m_radiance_miss_group
    ));

    memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = nullptr;  // NULL miss program for occlusion rays
    miss_prog_group_desc.miss.entryFunctionName = nullptr;
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
      scene.context(),
      &miss_prog_group_desc,
      1,                             // num program groups
      &program_group_options,
      log,
      &sizeof_log,
      &m_occlusion_miss_group
    ));
  }

  //
  // Hit group
  //
  {
    // Associate the shader selected in the command line with illum 0, 1, and 2
    OptixProgramGroup m_radiance_hit_group = createShader(1, shadername);
    setShader(0, m_radiance_hit_group);
    setShader(2, m_radiance_hit_group);
    createShader(3, "mirror");            // associate the mirror shader with illum 3
    createShader(4, "transparent");       // associate the transparent shader with illum 4
    createShader(5, "glossy");            // associate the glossy shader with illum 5
    createShader(11, "metal");            // associate the metal shader with illum 11
    createShader(30, "holdout");          // associate the holdout shader with illum 30

    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH = m_ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
      scene.context(),
      &hit_prog_group_desc,
      1,                             // num program groups
      &program_group_options,
      log,
      &sizeof_log,
      &m_occlusion_hit_group
    ));
  }
}

void ObjScene::createPipeline()
{
  OptixProgramGroup program_groups[] =
  {
      m_raygen_prog_group,
      m_radiance_miss_group,
      m_occlusion_miss_group,
      getShader(1),
      m_occlusion_hit_group,
  };

  OptixPipelineLinkOptions pipeline_link_options = {};
  pipeline_link_options.maxTraceDepth = MAX_DEPTH;
  pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK_LOG(optixPipelineCreate(
    scene.context(),
    &m_pipeline_compile_options,
    &pipeline_link_options,
    program_groups,
    sizeof(program_groups)/sizeof(program_groups[0]),
    log,
    &sizeof_log,
    &m_pipeline
  ));
}


void ObjScene::createSBT()
{
  {
    const size_t raygen_record_size = sizeof(EmptyRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_sbt.raygenRecord), raygen_record_size));

    EmptyRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(
      reinterpret_cast<void*>(m_sbt.raygenRecord),
      &rg_sbt,
      raygen_record_size,
      cudaMemcpyHostToDevice
    ));
  }

  {
    const unsigned int ray_type_count = RAY_TYPE_COUNT;
    const size_t miss_record_size = sizeof(EmptyRecord);
    CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&m_sbt.missRecordBase),
      miss_record_size*ray_type_count
    ));

    vector<EmptyRecord> ms_sbt(ray_type_count);
    OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_miss_group, &ms_sbt[RAY_TYPE_RADIANCE]));
    OPTIX_CHECK(optixSbtRecordPackHeader(m_occlusion_miss_group, &ms_sbt[RAY_TYPE_OCCLUSION]));

    CUDA_CHECK(cudaMemcpy(
      reinterpret_cast<void*>(m_sbt.missRecordBase),
      &ms_sbt[0],
      miss_record_size*ray_type_count,
      cudaMemcpyHostToDevice
    ));
    m_sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    m_sbt.missRecordCount = ray_type_count;
  }

  {
    vector<HitGroupRecord> hitgroup_records;
    for(const auto mesh : scene.meshes())
    {
      for(size_t i = 0; i < mesh->material_idx.size(); ++i)
      {
        HitGroupRecord rec = {};
        const int32_t mat_idx = mesh->material_idx[i];
        if(mat_idx >= 0)
        {
          rec.data.mtl_inside = m_materials[mat_idx];
          if(rec.data.mtl_inside.opposite >= 0)
            rec.data.mtl_outside = m_materials[rec.data.mtl_inside.opposite];
          else
            rec.data.mtl_outside = MtlData();
        }
        else
        {
          rec.data.mtl_inside = MtlData();
          rec.data.mtl_outside = MtlData();
        }
        OptixProgramGroup m_radiance_hit_group = getShader(rec.data.mtl_inside.illum);
        OPTIX_CHECK(optixSbtRecordPackHeader(m_radiance_hit_group, &rec));
        rec.data.geometry.type = GeometryData::TRIANGLE_MESH;
        rec.data.geometry.triangle_mesh.positions = mesh->positions[i];
        rec.data.geometry.triangle_mesh.normals = mesh->normals[i];
        rec.data.geometry.triangle_mesh.texcoords = mesh->texcoords[i];
        rec.data.geometry.triangle_mesh.indices = mesh->indices[i];
        hitgroup_records.push_back(rec);

        OPTIX_CHECK(optixSbtRecordPackHeader(m_occlusion_hit_group, &rec));
        hitgroup_records.push_back(rec);
      }
    }

    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&m_sbt.hitgroupRecordBase),
      hitgroup_record_size*hitgroup_records.size()
    ));
    CUDA_CHECK(cudaMemcpy(
      reinterpret_cast<void*>(m_sbt.hitgroupRecordBase),
      hitgroup_records.data(),
      hitgroup_record_size*hitgroup_records.size(),
      cudaMemcpyHostToDevice
    ));

    m_sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>(hitgroup_record_size);
    m_sbt.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_records.size());
  }
}

OptixProgramGroup ObjScene::createShader(int illum, string name)
{
  OptixProgramGroupOptions program_group_options = {};
  char log[2048];
  size_t sizeof_log = sizeof(log);
  string shader = "__closesthit__" + name;
  OptixProgramGroup m_radiance_hit_group = 0;
  OptixProgramGroupDesc hit_prog_group_desc = {};
  hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hit_prog_group_desc.hitgroup.moduleCH = m_ptx_module;
  hit_prog_group_desc.hitgroup.entryFunctionNameCH = shader.c_str();
  sizeof_log = sizeof(log);
  OPTIX_CHECK_LOG(optixProgramGroupCreate(
    scene.context(),
    &hit_prog_group_desc,
    1,                             // num program groups
    &program_group_options,
    log,
    &sizeof_log,
    &m_radiance_hit_group
  ));
  setShader(illum, m_radiance_hit_group);
  return m_radiance_hit_group;
}

void ObjScene::setShader(int illum, OptixProgramGroup closest_hit_program)
{
  if(illum < 0)
  {
    cerr << "Error: Negative identification numbers are not supported for illumination models." << endl;
    return;
  }
  while(illum >= static_cast<int>(shaders.size()))
    shaders.push_back(0);
  shaders[illum] = closest_hit_program;
}

OptixProgramGroup ObjScene::getShader(int illum)
{
  OptixProgramGroup shader = 0;
  if(illum >= 0 && illum < static_cast<int>(shaders.size()))
    shader = shaders[illum];
  
  if(!shader)
  {
    cerr << "Warning: An object uses a material with an unsupported illum identifier. Using the default shader instead." << endl;
    shader = shaders[0];
  }
  return shader;
}

void ObjScene::loadObjs()
{
  int mtl_count = 0;
  int tex_count = use_envmap ? 1 : 0;
  for(string filename : filenames)
  {
    scanMeshObj(filename);

    for(tinyobj::material_t& mtl : obj_materials)
    {
      MtlData m_mtl;
      m_mtl.rho_d = make_float3(mtl.diffuse[0], mtl.diffuse[1], mtl.diffuse[2]);
      m_mtl.rho_s = make_float3(mtl.specular[0], mtl.specular[1], mtl.specular[2]);
      m_mtl.emission = make_float3(mtl.ambient[0], mtl.ambient[1], mtl.ambient[2]);
      m_mtl.shininess = mtl.shininess;
      m_mtl.ior = mtl.ior;
      m_mtl.illum = mtl.illum;
      if(!mtl.diffuse_texname.empty())
      {
        string path;
        size_t idx = filename.find_last_of("/\\");
        if(idx < filename.length())
          path = filename.substr(0, idx + 1);

        ImageBuffer img = loadImage((path + mtl.diffuse_texname).c_str());
        if(img.pixel_format != UNSIGNED_BYTE4)
          cerr << "Texture image with unknown pixel format: " << mtl.diffuse_texname << endl;
        else
        {
          cout << "Loaded texture image " << mtl.diffuse_texname << endl;
          scene.addImage(img.width, img.height, 8, 4, img.data);
          scene.addSampler(cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, tex_count);
          m_mtl.base_color_tex = scene.getSampler(tex_count++);
        }
      }
      m_materials.push_back(m_mtl);
      mtl_names.push_back(mtl.name);
    }
    for(vector<tinyobj::shape_t>::const_iterator it = obj_shapes.begin(); it < obj_shapes.end(); ++it)
    {
      const tinyobj::shape_t& shape = *it;
      CUdeviceptr buffer;
      auto mesh = std::make_shared<Scene::MeshGroup>();
      scene.addMesh(mesh);
      mesh->name = shape.name;
      {
        BufferView<unsigned int> buffer_view;
        scene.addBuffer(shape.mesh.indices.size()*sizeof(unsigned int), reinterpret_cast<const void*>(&shape.mesh.indices[0]));
        buffer = scene.getBuffer(bufidx++);
        buffer_view.data = buffer;
        buffer_view.byte_stride = 0;
        buffer_view.count = static_cast<uint32_t>(shape.mesh.indices.size());
        buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(unsigned int));
        mesh->indices.push_back(buffer_view);
      }
      {
        BufferView<float3> buffer_view;
        scene.addBuffer(shape.mesh.positions.size()*sizeof(float), reinterpret_cast<const void*>(&shape.mesh.positions[0]));
        buffer = scene.getBuffer(bufidx++);
        buffer_view.data = buffer;
        buffer_view.byte_stride = 0;
        buffer_view.count = static_cast<uint32_t>(shape.mesh.positions.size()/3);
        buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
        mesh->positions.push_back(buffer_view);
      }
      {
        BufferView<float3> buffer_view;
        if(shape.mesh.normals.size() > 0)
        {
          scene.addBuffer(shape.mesh.normals.size()*sizeof(float), reinterpret_cast<const void*>(&shape.mesh.normals[0]));
          buffer = scene.getBuffer(bufidx++);
          buffer_view.data = buffer;
          buffer_view.byte_stride = 0;
          buffer_view.count = static_cast<uint32_t>(shape.mesh.normals.size()/3);
          buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
        }
        mesh->normals.push_back(buffer_view);
      }
      {
        BufferView<float2> buffer_view;
        if(shape.mesh.texcoords.size() > 0)
        {
          scene.addBuffer(shape.mesh.texcoords.size()*sizeof(float), reinterpret_cast<const void*>(&shape.mesh.texcoords[0]));
          buffer = scene.getBuffer(bufidx++);
          buffer_view.data = buffer;
          buffer_view.byte_stride = 0;
          buffer_view.count = static_cast<uint32_t>(shape.mesh.texcoords.size()/2);
          buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float2));
        }
        mesh->texcoords.push_back(buffer_view);
      }
      mesh->material_idx.push_back(shape.mesh.material_ids[0] + mtl_count);
      mesh->transform = get_object_transform(filename);
      cerr << "\t\tNum triangles: " << mesh->indices.back().count/3 << endl;

      Surface surface;
      surface.indices.resize(shape.mesh.indices.size()/3);
      copy(shape.mesh.indices.begin(), shape.mesh.indices.end(), &surface.indices.front().x);
      surface.positions.resize(shape.mesh.positions.size()/3);
      for(unsigned int i = 0; i < surface.positions.size(); ++i)
      {
        float4 pos = make_float4(shape.mesh.positions[i*3], shape.mesh.positions[i*3 + 1], shape.mesh.positions[i*3 + 2], 1.0f);
        surface.positions[i] = make_float3(mesh->transform*pos);
      }
      //copy(shape.mesh.positions.begin(), shape.mesh.positions.end(), &surface.positions.front().x);
      if(shape.mesh.normals.size() > 0)
      {
        surface.normals.resize(shape.mesh.normals.size()/3);
        for(unsigned int i = 0; i < surface.normals.size(); ++i)
        {
          float4 normal = make_float4(shape.mesh.positions[i*3], shape.mesh.positions[i*3 + 1], shape.mesh.positions[i*3 + 2], 0.0f);
          surface.normals[i] = make_float3(mesh->transform*normal);
        }
        //copy(shape.mesh.normals.begin(), shape.mesh.normals.end(), &surface.normals.front().x);
      }
      surface.no_of_faces = static_cast<unsigned int>(surface.indices.size());
      surfaces.push_back(surface);

      mesh->object_aabb.invalidate();
      for(unsigned int i = 0; i < shape.mesh.positions.size()/3; ++i)
        mesh->object_aabb.include(make_float3(shape.mesh.positions[i*3], shape.mesh.positions[i*3 + 1], shape.mesh.positions[i*3 + 2]));
      mesh->world_aabb = mesh->object_aabb;
      mesh->world_aabb.transform(mesh->transform);
    }
    mtl_count += static_cast<int>(obj_materials.size());
    obj_materials.clear();
    obj_shapes.clear();
  }
}

void ObjScene::scanMeshObj(string m_filename)
{
  int32_t num_triangles = 0;
  int32_t num_vertices = 0;
  int32_t num_materials = 0;
  bool has_normals = false;
  bool has_texcoords = false;

  if(obj_shapes.empty())
  {
    std::string err;
    bool ret = tinyobj::LoadObj(
      obj_shapes,
      obj_materials,
      err,
      m_filename.c_str(),
      m_filename.substr(0, m_filename.find_last_of("\\/") + 1).c_str()
    );

    if(!err.empty())
      cerr << err << endl;

    if(!ret)
      throw runtime_error("MeshLoader: " + err);
  }

  //
  // Iterate over all shapes and sum up number of vertices and triangles
  //
  uint64_t num_groups_with_normals = 0;
  uint64_t num_groups_with_texcoords = 0;
  for(vector<tinyobj::shape_t>::const_iterator it = obj_shapes.begin(); it < obj_shapes.end(); ++it)
  {
    const tinyobj::shape_t& shape = *it;

    num_triangles += static_cast<int32_t>(shape.mesh.indices.size())/3;
    num_vertices += static_cast<int32_t>(shape.mesh.positions.size())/3;

    if(!shape.mesh.normals.empty())
      ++num_groups_with_normals;

    if(!shape.mesh.texcoords.empty())
      ++num_groups_with_texcoords;
  }

  //
  // We ignore normals and texcoords unless they are present for all shapes
  //
  if(num_groups_with_normals != 0)
  {
    if(num_groups_with_normals != obj_shapes.size())
      cerr << "MeshLoader - WARNING: mesh '" << m_filename
           << "' has normals for some groups but not all.  "
           << "Ignoring all normals." << endl;
    else
      has_normals = true;
  }
  if(num_groups_with_texcoords != 0)
  {
    if(num_groups_with_texcoords != obj_shapes.size())
      cerr << "MeshLoader - WARNING: mesh '" << m_filename
           << "' has texcoords for some groups but not all.  "
           << "Ignoring all texcoords." << endl;
    else
      has_texcoords = true;
  }
  num_materials = (int32_t)m_materials.size();
}

void ObjScene::add_default_light()
{
  // The radiance of a directional source modeling the Sun should be equal
  // to the irradiance at the surface of the Earth.
  // We convert radiance to irradiance at the surface of the Earth using the
  // solid angle 6.74e-5 subtended by the solar disk as seen from Earth.

  // Default directional light
  vector<Directional> dir_lights(1);
  dir_lights[0].emission = light_rad;
  dir_lights[0].direction = normalize(light_dir);

  if(launch_params.lights.count == 0)
  {
    launch_params.lights.count = static_cast<uint32_t>(dir_lights.size());
    CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&launch_params.lights.data),
      dir_lights.size()*sizeof(Directional)
    ));
  }
  CUDA_CHECK(cudaMemcpy(
    reinterpret_cast<void*>(launch_params.lights.data),
    dir_lights.data(),
    dir_lights.size()*sizeof(Directional),
    cudaMemcpyHostToDevice
  ));
}

unsigned int ObjScene::extract_area_lights()
{
  vector<uint2> lights;
  auto& meshes = scene.meshes();
  int mesh_idx = 0;
  for(auto mesh : meshes)
  {
    for(unsigned int j = 0; j < mesh->material_idx.size(); ++j)
    {
      int mtl_idx = mesh->material_idx[j];
      const MtlData& mtl = m_materials[mtl_idx];
      bool emissive = false;
      for(unsigned int k = 0; k < 3; ++k)
        emissive = emissive || *(&mtl.emission.x + k) > 0.0f;
      if(emissive)
        lights.push_back(make_uint2(mesh_idx, mtl_idx));
    }
    ++mesh_idx;
  }
  Surface lightsurf;
  vector<float3> emission;
  for(unsigned int j = 0; j < lights.size(); ++j)
  {
    uint2 light = lights[j];
    auto mesh = meshes[light.x];
    const Surface& surface = surfaces[light.x];
    unsigned int no_of_verts = static_cast<unsigned int>(lightsurf.positions.size());
    lightsurf.positions.insert(lightsurf.positions.end(), surface.positions.begin(), surface.positions.end());
    lightsurf.normals.insert(lightsurf.normals.end(), surface.positions.begin(), surface.positions.end());
    lightsurf.indices.insert(lightsurf.indices.end(), surface.indices.begin(), surface.indices.end());
    if(surface.normals.size() > 0)
      for(unsigned int k = no_of_verts; k < lightsurf.normals.size(); ++k)
        lightsurf.normals[k] = surface.normals[k - no_of_verts];
    for(unsigned int k = lightsurf.no_of_faces; k < lightsurf.indices.size(); ++k)
    {
      lightsurf.indices[k] += make_uint3(no_of_verts);
      if(surface.normals.size() == 0)
      {
        uint3 face = lightsurf.indices[k];
        float3 p0 = lightsurf.positions[face.x];
        float3 a = lightsurf.positions[face.y] - p0;
        float3 b = lightsurf.positions[face.z] - p0;
        lightsurf.normals[face.x] = lightsurf.normals[face.y] = lightsurf.normals[face.z] = normalize(cross(a, b));
      }
    }
    emission.insert(emission.end(), surface.no_of_faces, m_materials[light.y].emission);
    lightsurf.no_of_faces += surface.no_of_faces;
  }
  {
    BufferView<uint3> buffer_view;
    scene.addBuffer(lightsurf.indices.size()*sizeof(uint3), reinterpret_cast<const void*>(&lightsurf.indices[0]));
    buffer_view.data = scene.getBuffer(bufidx++);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(lightsurf.indices.size());
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(uint3));
    launch_params.light_idxs = buffer_view;
  }
  {
    BufferView<float3> buffer_view;
    scene.addBuffer(lightsurf.positions.size()*sizeof(float3), reinterpret_cast<const void*>(&lightsurf.positions[0]));
    buffer_view.data = scene.getBuffer(bufidx++);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(lightsurf.positions.size());
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
    launch_params.light_verts = buffer_view;
  }
  {
    BufferView<float3> buffer_view;
    scene.addBuffer(lightsurf.normals.size()*sizeof(float3), reinterpret_cast<const void*>(&lightsurf.normals[0]));
    buffer_view.data = scene.getBuffer(bufidx++);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(lightsurf.normals.size());
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
    launch_params.light_norms = buffer_view;
  }
  {
    BufferView<float3> buffer_view;
    scene.addBuffer(emission.size()*sizeof(float3), reinterpret_cast<const void*>(&emission[0]));
    buffer_view.data = scene.getBuffer(bufidx++);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(emission.size());
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float3));
    launch_params.light_emission = buffer_view;
  }
  float surface_area = 0.0f;
  vector<float> face_areas(lightsurf.no_of_faces);
  vector<float> face_area_cdf(lightsurf.no_of_faces);
  for(unsigned int i = 0; i < lightsurf.no_of_faces; ++i)
  {
    uint3 face = lightsurf.indices[i];
    float3 p0 = lightsurf.positions[face.x];
    float3 a = lightsurf.positions[face.y] - p0;
    float3 b = lightsurf.positions[face.z] - p0;
    face_areas[i] = 0.5f*length(cross(a, b));
    face_area_cdf[i] = surface_area + face_areas[i];
    surface_area += face_areas[i];
  }
  if(surface_area > 0.0f)
    for(unsigned int i = 0; i < lightsurf.no_of_faces; ++i)
      face_area_cdf[i] /= surface_area;
  launch_params.light_area = surface_area;
  {
    BufferView<float> buffer_view;
    scene.addBuffer(face_area_cdf.size()*sizeof(float), reinterpret_cast<const void*>(&face_area_cdf[0]));
    buffer_view.data = scene.getBuffer(bufidx++);
    buffer_view.byte_stride = 0;
    buffer_view.count = static_cast<uint32_t>(face_area_cdf.size());
    buffer_view.elmt_byte_size = static_cast<uint16_t>(sizeof(float));
    launch_params.light_face_area_cdf = buffer_view;
  }
  return static_cast<unsigned int>(lights.size());
}

Matrix4x4 ObjScene::get_object_transform(string filename) const
{
  size_t idx = filename.find_last_of("\\/") + 1;
  if(idx < filename.length())
  {
    if(filename.compare(idx, 7, "cornell") == 0)
      return Matrix4x4::scale(make_float3(0.025f))*Matrix4x4::rotate(M_PIf, make_float3(0.0f, 1.0f, 0.0f));
    else if(filename.compare(idx, 6, "dragon") == 0)
      return Matrix4x4::rotate(-M_PI_2f, make_float3(1.0, 0.0, 0.0));
    else if(filename.compare(idx, 5, "bunny") == 0)
      return Matrix4x4::translate(make_float3(-3.0f, -0.84f, -8.0f))*Matrix4x4::scale(make_float3(25.0f))*Matrix4x4::rotate(0.02f, make_float3(1.0f, 0.0f, 0.0f));
    else if(filename.compare(idx, 12, "justelephant") == 0)
      return Matrix4x4::translate(make_float3(-10.0f, 3.0f, -2.0f))*Matrix4x4::rotate(0.5f, make_float3(0.0f, 1.0f, 0.0f));
    else if(filename.compare(idx, 10, "glass_wine") == 0)
      return Matrix4x4::scale(make_float3(5.0f));
  }
  return Matrix4x4::identity();
}
