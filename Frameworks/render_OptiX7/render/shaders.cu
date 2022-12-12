#include <optix.h>

#include <cuda/LocalGeometry.h>
#include <cuda/helpers.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>

#include "structs.h"
#include "trace.h"
#include "cdf_bsearch.h"
#include "sampler.h"
#include "fresnel.h"
#include "microfacet.h"

extern "C" {
  __constant__ LaunchParams launch_params;
}

#include "envmap.h"
#include "AreaLight.h"

#define DIRECT
#define INDIRECT

__device__ __inline__ uchar4 make_rgba(const float3& color)
{
  float3 c = clamp(color, 0.0f, 1.0f);
  return make_uchar4(quantizeUnsigned8Bits(c.x), quantizeUnsigned8Bits(c.y), quantizeUnsigned8Bits(c.z), 255u);
}

extern "C" __global__ void __raygen__pinhole()
{
  const LaunchParams& lp = launch_params;
  const uint3 launch_idx = optixGetLaunchIndex();
  const uint3 launch_dims = optixGetLaunchDimensions();
  const unsigned int frame = lp.subframe_index;
  const unsigned int image_idx = launch_idx.y*launch_dims.x + launch_idx.x;
  unsigned int t = tea<16>(image_idx, frame);

  // Generate camera ray (the center of each pixel is at (0.5, 0.5))
  //const float2 jitter = subframe_index == 0 ? make_float2(0.5f, 0.5f) : make_float2(rnd(t), rnd(t));
  const float2 jitter = make_float2(rnd(t), rnd(t));
  const float2 idx = make_float2(launch_idx.x, launch_idx.y);
  const float2 res = make_float2(launch_dims.x, launch_dims.y);
  const float2 ip_coords = (idx + jitter)/res*2.0f - 1.0f;
  const float3 direction = normalize(ip_coords.x*lp.U + ip_coords.y*lp.V + lp.W);

  // Trace camera ray
  PayloadRadiance payload;
  payload.result = make_float3(0.0f);
  payload.depth = 0;
  payload.seed = t;
  payload.emit = 1;
  traceRadiance(lp.handle, lp.eye, direction, 0.0f, 1.0e16f, &payload);

  // Progressive update of image
  float3 curr_sum = make_float3(lp.accum_buffer[image_idx])*static_cast<float>(frame);
  float3 accum_color = (payload.result + curr_sum)/static_cast<float>(frame + 1);

  lp.accum_buffer[image_idx] = make_float4(accum_color, 1.0f);
  //lp.frame_buffer[image_idx] = make_color(accum_color);  // use to output sRGB images
  lp.frame_buffer[image_idx] = make_rgba(accum_color);  // use to output RGB images (no gamma)
}


extern "C" __global__ void __miss__constant_radiance()
{
#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  prd->result = launch_params.miss_color;
#else
  setPayloadResult(launch_params.miss_color);
#endif
}


extern "C" __global__ void __miss__envmap_radiance()
{
  const float3 ray_dir = optixGetWorldRayDirection();

#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  prd->result = env_lookup(ray_dir);
#else
  setPayloadResult(env_lookup(ray_dir));
#endif
}


extern "C" __global__ void __closesthit__occlusion()
{
  setPayloadOcclusion(true);
}


extern "C" __global__ void __closesthit__normals()
{
  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry);

  float3 result = normalize(geom.N)*0.5f + 0.5f;
#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  prd->result = result;
#else
  setPayloadResult(result);
#endif
}


extern "C" __global__ void __closesthit__basecolor()
{
  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry);

  // Retrieve material data
  const float3& emission = hit_group_data->mtl_inside.emission;
  float3 rho_d = hit_group_data->mtl_inside.base_color_tex
    ? make_float3(tex2D<float4>(hit_group_data->mtl_inside.base_color_tex, geom.UV.x, geom.UV.y))
    : hit_group_data->mtl_inside.rho_d;

  float3 result = rho_d + emission;
#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  prd->result = result;
#else
  setPayloadResult(result);
#endif
}


extern "C" __global__ void __closesthit__directional()
{
  const LaunchParams& lp = launch_params;
#ifdef INDIRECT
#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  unsigned int depth = prd->depth;
  unsigned int& t = prd->seed;
#else
  unsigned int depth = getPayloadDepth();
  unsigned int t = getPayloadSeed();
#endif
  if(depth > lp.max_depth)
    return;
#endif
  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry);

  // Retrieve material data
  const float3& emission = hit_group_data->mtl_inside.emission;
  float3 rho_d = hit_group_data->mtl_inside.base_color_tex
    ? make_float3(tex2D<float4>(hit_group_data->mtl_inside.base_color_tex, geom.UV.x, geom.UV.y))
    : hit_group_data->mtl_inside.rho_d;

  // Retrieve hit info
  const float3& x = geom.P;
  const float3 n = normalize(geom.N);

  // Implement Lambertian reflection here, include shadow rays.
  //
  // Output: 
  // result        (payload: result is the reflected radiance)
  //
  // Relevant data fields that are available (see above):
  // rho_d         (difuse reflectance of the material)
  // x             (position where the ray hit the material)
  // n             (normal where the ray hit the material)
  // lp.lights     (array of directional light sources)
  // lp.handle     (spatial data structure handle for tracing new rays)
  //
  // Hint: Use the function traceOcclusion to trace a shadow ray.
  float3 result = emission;
  const float tmin = 1.0e-4f;
  const float tmax = 1.0e16f; 



#ifdef PASS_PAYLOAD_POINTER
#ifndef INDIRECT
  PayloadRadiance* prd = getPayload();
#endif
  prd->result = result;
#else
  setPayloadResult(result);
#endif
  }


extern "C" __global__ void __closesthit__arealight()
{
#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  unsigned int depth = prd->depth;
  unsigned int& t = prd->seed;
  unsigned int emit = prd->emit;
#else
  unsigned int depth = getPayloadDepth();
  unsigned int t = getPayloadSeed();
  unsigned int emit = getPayloadEmit();
#endif
#ifdef INDIRECT
  const LaunchParams& lp = launch_params;
  if(depth > lp.max_depth)
    return;
#endif
  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry);

  // Retrieve material data
  const float3& emission = hit_group_data->mtl_inside.emission;
  float3 rho_d = hit_group_data->mtl_inside.base_color_tex
    ? make_float3(tex2D<float4>(hit_group_data->mtl_inside.base_color_tex, geom.UV.x, geom.UV.y))
    : hit_group_data->mtl_inside.rho_d;

  // Retrieve ray and hit info
  const float3 ray_dir = optixGetWorldRayDirection();
  const float3& x = geom.P;
  const float3 n = normalize(geom.N)*copysignf(1.0f, -dot(geom.N, ray_dir));
  float3 result = emit ? emission : make_float3(0.0f);
  const float tmin = 1.0e-4f;
  const float tmax = 1.0e16f;

  // Compute direct illumination
  float3 wi;
  float3 Le;
  float dist;
  sample(x, wi, Le, dist, t);

  // Implement Lambertian reflection here, include shadow rays.
  //
  // Output: 
  // result        (payload: result is the reflected radiance)
  //
  // Relevant data fields that are available (see above):
  // rho_d         (difuse reflectance of the material)
  // x             (position where the ray hit the material)
  // n             (normal where the ray hit the material)
  // wi            (sampled direction toward the light)
  // Le            (emitted radiance received from the direction w_i)
  // dist          (distance to the sampled position on the light source)
  //
  // Hint: Implement the function sample_center(...) or the function sample(...) in AreaLight.h first.



#ifdef PASS_PAYLOAD_POINTER
  prd->result = result;
#else
  setPayloadResult(result);
#endif
}


extern "C" __global__ void __closesthit__holdout()
{
  const LaunchParams& lp = launch_params;
#ifdef INDIRECT
#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  unsigned int depth = prd->depth;
  unsigned int& t = prd->seed;
#else
  unsigned int depth = getPayloadDepth();
  unsigned int t = getPayloadSeed();
#endif
  if(depth > lp.max_depth)
    return;
#endif
  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry);

  // Retrieve material data
  const float3& emission = hit_group_data->mtl_inside.emission;
  const float3 ray_dir = optixGetWorldRayDirection();
  float3 rho_d = env_lookup(ray_dir);

  // Retrieve hit info
  const float3& x = geom.P;
  const float3 n = normalize(geom.N);
  float3 result = emission;
  const float tmin = 1.0e-4f;
  const float tmax = 1.0e16f;



#ifdef PASS_PAYLOAD_POINTER
#ifndef INDIRECT
  PayloadRadiance* prd = getPayload();
#endif
  prd->result = result;
#else
  setPayloadResult(result);
#endif
}


extern "C" __global__ void __closesthit__mirror()
{
#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  unsigned int depth = prd->depth;
  unsigned int& t = prd->seed;
#else
  unsigned int depth = getPayloadDepth();
  unsigned int t = getPayloadSeed();
#endif
  const LaunchParams& lp = launch_params;
  if(depth > lp.max_depth)
    return;

  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry);

  // Retrieve ray and hit info
  const float3 ray_dir = optixGetWorldRayDirection();
  const float3& x = geom.P;
  const float3 n = normalize(geom.N);

  // Implement mirror reflection here.
  //
  // Output: 
  // result        (payload: result is the reflected radiance)
  //
  // Relevant data fields that are available (see above):
  // x             (position where the ray hit the material)
  // n             (normal where the ray hit the material)
  // ray_dir       (direction of the ray)
  //
  // Hint: Make a new PayloadRadiance and use the function traceRadiance
  //       to trace a new radiance ray.
  PayloadRadiance payload;



#ifdef PASS_PAYLOAD_POINTER
  prd->result = payload.result;
#else
  setPayloadResult(payload.result);
#endif
}

extern "C" __global__ void __closesthit__transparent()
{
#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  unsigned int depth = prd->depth;
  unsigned int& t = prd->seed;
#else
  unsigned int depth = getPayloadDepth();
  unsigned int t = getPayloadSeed();
#endif
  const LaunchParams& lp = launch_params;
  if(depth > lp.max_depth)
    return;

  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry);

  // Retrieve material data
  const float ior1 = hit_group_data->mtl_outside.ior;
  const float ior2 = hit_group_data->mtl_inside.ior;

  // Retrieve ray and hit info
  const float3 ray_dir = optixGetWorldRayDirection();
  const float3& x = geom.P;
  float3 n = normalize(geom.N);

  // Implement reflection and refraction in transparent medium.
  //
  // Output: 
  // result        (payload: result is the reflected radiance)
  //
  // Relevant data fields that are available (see above):
  // x             (position where the ray hit the material)
  // n             (normal where the ray hit the material)
  // ray_dir       (direction of the ray)
  //
  // Hint: (a) Use Russian roulette to choose reflection or refraction and
  //       make sure that you handle total internal reflection.
  //       (b) The Fresnel equations are available by means of the function
  //       fresnel_R, which is implemented in fresnel.h.
  float3 result = make_float3(0.0f);



#ifdef PASS_PAYLOAD_POINTER
  prd->result = result;
#else
  setPayloadResult(result);
#endif
}

extern "C" __global__ void __closesthit__glossy()
{
#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  unsigned int depth = prd->depth;
  unsigned int& t = prd->seed;
#else
  unsigned int depth = getPayloadDepth();
  unsigned int t = getPayloadSeed();
#endif
  const LaunchParams& lp = launch_params;
  if(depth > lp.max_depth)
    return;

  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry);

  // Retrieve material data
  const float ior = hit_group_data->mtl_inside.ior;
  const float s = 1.0f/hit_group_data->mtl_inside.shininess;

  // Retrieve ray and hit info
  const float3 ray_dir = optixGetWorldRayDirection();
  const float3& x = geom.P;
  float3 n = normalize(geom.N);

  // Rough reflection/refraction based on a normal distribution
  float3 result = make_float3(0.0f);


#ifdef PASS_PAYLOAD_POINTER
  prd->result = result;
#else
  setPayloadResult(result);
#endif
}

extern "C" __global__ void __closesthit__metal()
{
#ifdef PASS_PAYLOAD_POINTER
  PayloadRadiance* prd = getPayload();
  unsigned int depth = prd->depth;
  unsigned int& t = prd->seed;
#else
  unsigned int depth = getPayloadDepth();
  unsigned int t = getPayloadSeed();
#endif
  const LaunchParams& lp = launch_params;
  if(depth > lp.max_depth)
    return;

  const HitGroupData* hit_group_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  const LocalGeometry geom = getLocalGeometry(hit_group_data->geometry);

  // Retrieve ray and hit info
  const float3 ray_dir = optixGetWorldRayDirection();
  const float3& x = geom.P;
  const float3 n = normalize(geom.N);

  // Implement mirror reflection here.
  //
  // Output: 
  // result        (payload: result is the reflected radiance)
  //
  // Relevant data fields that are available (see above):
  // x             (position where the ray hit the material)
  // n             (normal where the ray hit the material)
  // ray_dir       (direction of the ray)
  //
  // Hint: Make a new PayloadRadiance and use the function traceRadiance
  //       to trace a new radiance ray.
  float3 result = make_float3(0.0f);



#ifdef PASS_PAYLOAD_POINTER
  prd->result = result;
#else
  setPayloadResult(result);
#endif
}

