// 02562 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2020
// Copyright (c) DTU Compute 2020

#ifndef AREALIGHT_H
#define AREALIGHT_H

#include <optix.h>
#include <cuda/random.h>
#include "cdf_bsearch.h"
#include "sampler.h"
#include "structs.h"


__device__ __inline__ uint3 get_light_triangle(unsigned int idx, float3& v0, float3& v1, float3& v2)
{
  const LaunchParams& lp = launch_params;
  const uint3& face = lp.light_idxs[idx];
  v0 = lp.light_verts[face.x];
  v1 = lp.light_verts[face.y];
  v2 = lp.light_verts[face.z];
  return face;
}

__device__ __inline__ void sample_center(const float3& pos, float3& dir, float3& L, float& dist)
{
  // Compute output given the following information.
  //
  // Input:  pos    (observed surface position in the scene)
  //
  // Output: dir    (direction toward the light)
  //         L      (radiance received from the direction dir)
  //         dist   (distance to the sampled position on the light source)
  //
  // Relevant data fields that are available (see above):
  // lp.light_verts    (vertex positions for the indexed face set representing the light source)
  // lp.light_norms    (vertex normals for the indexed face set representing the light source)
  // lp.light_idxs     (vertex indices for each triangle in the indexed face set)
  // lp.light_emission (radiance emitted by each triangle of the light source)
  //
  // Hint: (a) Find the face normal for each triangle (using the function get_light_triangle) and
  //        use these to add up triangle areas and find the average normal.
  //       (b) OptiX includes a function normalize(v) which returns the 
  //       vector v normalized to unit length.


}

__device__ __inline__ void sample(const float3& pos, float3& dir, float3& L, float& dist, unsigned int& t)
{
  // Compute output given the following information.
  //
  // Input:  pos    (observed surface position in the scene)
  //
  // Output: dir    (direction toward the light)
  //         L      (radiance received from the direction dir)
  //         dist   (distance to the sampled position on the light source)
  //
  // Relevant data fields that are available (see above):
  // lp.light_verts         (vertex positions for the indexed face set representing the light source)
  // lp.light_norms         (vertex normals for the indexed face set representing the light source)
  // lp.light_idxs          (vertex indices for each triangle in the indexed face set)
  // lp.light_emission      (radiance emitted by each triangle in the indexed face set)
  // lp.light_area          (total surface area of light source)
  // lp.light_face_area_cdf (discrete cdf for sampling a triangle index using binary search)
  //
  // Hint: (a) Get random numbers using rnd(t).
  //       (b) There is a cdf_bsearch function available for doing binary search.
  const LaunchParams& lp = launch_params;


}

#endif // AREALIGHT_H