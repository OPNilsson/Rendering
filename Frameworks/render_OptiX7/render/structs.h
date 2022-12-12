#pragma once

#include <optix.h>
#include <cuda/GeometryData.h>
#include <texture_types.h>
#include "Directional.h"

//#define PASS_PAYLOAD_POINTER

struct LaunchParams
{
  unsigned int             subframe_index;
  float4*                  accum_buffer;
  uchar4*                  frame_buffer;
  unsigned int             max_depth;

  float3                   eye;
  float3                   U;
  float3                   V;
  float3                   W;

  BufferView<Directional>  lights;
  BufferView<float3>       light_verts;
  BufferView<float3>       light_norms;
  BufferView<uint3>        light_idxs;
  BufferView<float3>       light_emission;
  BufferView<float>        light_face_area_cdf;
  float                    light_area;

  cudaTextureObject_t      envmap;
  float3                   miss_color;
  OptixTraversableHandle   handle;
};

enum RayType
{
  RAY_TYPE_RADIANCE = 0,
  RAY_TYPE_OCCLUSION = 1,
  RAY_TYPE_COUNT = 2
};

struct PayloadRadiance
{
  float3 result;
  unsigned int depth;
  unsigned int seed;
  unsigned int emit;
};

#ifdef PASS_PAYLOAD_POINTER
const unsigned int NUM_PAYLOAD_VALUES = 2u; // pointer to payload
#else
const unsigned int NUM_PAYLOAD_VALUES = 6u; // no. of 32 bit fields in payload
#endif

struct PayloadOcclusion
{
};

struct MtlData
{
  float3 rho_d = { 1.0f, 1.0f, 1.0f };
  float3 rho_s = { 0.0f, 0.0f, 0.0f };
  float3 emission = { 0.0f, 0.0f, 0.0f };
  float shininess = 0.0f;
  float ior = 1.0f;
  int illum = 1;
  cudaTextureObject_t base_color_tex = 0;
  int opposite = -1;
};

struct HitGroupData
{
  GeometryData geometry;
  MtlData mtl_inside;
  MtlData mtl_outside;
};
