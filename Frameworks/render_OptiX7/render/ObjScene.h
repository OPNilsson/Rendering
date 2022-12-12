#pragma once

#include <string>
#include <vector>
#include <optix.h>
#include <sutil/Scene.h>
#include <sutil/CUDAOutputBuffer.h>

#include "structs.h"

class ObjScene
{
public:
  ObjScene(const std::vector<std::string>& obj_filenames,
           const std::string& shader_name, const std::string& env_filename,
           int32_t frame_width, int32_t frame_height,
           const float3& light_direction, const float3& light_emission)
    : filenames(obj_filenames), shadername(shader_name), envfile(env_filename),
      resize_dirty(false), minimized(false),
      width(frame_width), height(frame_height),
      light_dir(light_direction), light_rad(light_emission)
  { 
    if(shadername.empty())
      shadername = "normals";
    use_envmap = !envfile.empty();
  }

  ~ObjScene();

  void initScene();
  void initLaunchParams(const sutil::Scene& scene);
  void initCameraState();

  void handleCameraUpdate();
  void handleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer, int32_t w, int32_t h);
  void handleLightUpdate() { add_default_light(); }

  void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer);

  // Window resize state
  bool resize_dirty;
  bool minimized;

  // Camera state
  sutil::Camera camera;
  int32_t width;
  int32_t height;

  // Default light configuration
  float3 light_dir;
  float3 light_rad;

private:
  void loadObjs();
  void scanMeshObj(std::string m_filename);
  void add_default_light();
  unsigned int extract_area_lights();
  sutil::Matrix4x4 get_object_transform(std::string filename) const;
  void createPTXModule();
  void createProgramGroups();
  void createPipeline();
  void createSBT();

  OptixProgramGroup createShader(int illum, std::string name);
  void setShader(int illum, OptixProgramGroup closest_hit_program);
  OptixProgramGroup getShader(int illum);

  std::vector<MtlData> m_materials;
  std::vector<std::string> mtl_names;
  OptixShaderBindingTable m_sbt = {};
  OptixShaderBindingTable m_sample_sbt = {};
  OptixPipelineCompileOptions m_pipeline_compile_options = {};
  OptixPipeline m_pipeline = 0;
  OptixModule m_ptx_module = 0;
  OptixProgramGroup m_raygen_prog_group = 0;
  OptixProgramGroup m_sample_prog_group = 0;
  OptixProgramGroup m_radiance_miss_group = 0;
  OptixProgramGroup m_occlusion_miss_group = 0;
  OptixProgramGroup m_feeler_miss_group = 0;
  std::vector<OptixProgramGroup> shaders;
  OptixProgramGroup m_occlusion_hit_group = 0;
  OptixProgramGroup m_feeler_hit_group = 0;

  struct Surface
  {
    unsigned int no_of_faces = 0;
    std::vector<uint3> indices;
    std::vector<float3> positions;
    std::vector<float3> normals;
  };

  std::vector<std::string> filenames;
  std::vector<Surface> surfaces;
  std::string shadername;
  std::string envfile;  
  sutil::Scene scene;
  sutil::Aabb bbox;
  bool use_envmap;
};

