// 02562 Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <optix_world.h>
#include "mt_random.h"
#include "sampler.h"
#include "HitInfo.h"
#include "MCGlossy.h"

using namespace optix;

#ifndef M_1_PIf
#define M_1_PIf 0.31830988618379067154
#endif

float3 MCGlossy::shade(const Ray &r, HitInfo &hit, bool emit) const {
    if (hit.trace_depth >= max_depth)
        return make_float3(0.0f);

    float3 rho_d = get_diffuse(hit);
    float3 result = make_float3(0.0f);

    // Implement a path tracing shader here.
    //
    // Input:  r          (the ray that hit the material)
    //         hit        (info about the ray-surface intersection)
    //         emit       (passed on to Emission::shade)
    //
    // Return: radiance reflected to where the ray was coming from
    //
    // Relevant data fields that are available (see Mirror.h and HitInfo.h):
    // max_depth          (maximum trace depth)
    // tracer             (pointer to ray tracer)
    // hit.trace_depth    (number of surface interactions previously suffered by the ray)
    //
    // Hint: Use the function shade_new_ray(...) to pass a newly traced ray to
    //       the shader for the surface it hit.

    // Setup Monte Carlo variables
    float3 cos_theta = hit.geometric_normal;
    float3 pdf = cos_theta / M_PIf;
    float3 fr = rho_d / M_PIf;

    // Step 1: chose a new ray direction
    // Trance a new ray to see if it hit anything
    Ray new_ray = Ray(hit.position, sample_cosine_weighted(hit.shading_normal), 0, 0.001f, RT_DEFAULT_MAX);
    HitInfo new_hit;

    // Step 2: Trace ray to find point of intersection with the nearest surface
    if (tracer->trace_to_closest(new_ray, new_hit)) {
        new_hit.trace_depth++;
        result += shade_new_ray(new_ray, new_hit, false);
    }

    return rho_d * result + Phong::shade(r, hit, emit);;
}
