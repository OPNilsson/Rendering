// 02562 Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <optix_world.h>
#include "HitInfo.h"
#include "Phong.h"

using namespace optix;

#ifndef M_1_PIf
#define M_1_PIf 0.31830988618379067154
#endif

float3 Phong::shade(const Ray &r, HitInfo &hit, bool emit) const {
    float3 rho_d = get_diffuse(hit);
    float3 rho_s = get_specular(hit);
    float s = get_shininess(hit);

    // Implement Phong reflection here.
    //
    // Input:  r          (the ray that hit the material)
    //         hit        (info about the ray-surface intersection)
    //         emit       (passed on to Emission::shade)
    //
    // Return: radiance reflected to where the ray was coming from
    //
    // Relevant data fields that are available (see Lambertian.h, Ray.h, and above):
    // lights             (vector of pointers to the lights in the scene)
    // hit.position       (position where the ray hit the material)
    // hit.shading_normal (surface normal where the ray hit the material)
    // rho_d              (difuse reflectance of the material)
    // rho_s              (specular reflectance of the material)
    // s                  (shininess or Phong exponent of the material)
    //
    // Hint: Call the sample function associated with each light in the scene.

    // Sample variables
    float3 dir = make_float3(0.0f, 0.0f, 0.0f);
    float3 dir_reflected = make_float3(0.0f, 0.0f, 0.0f);
    float3 L_i = make_float3(0.0f, 0.0f, 0.0f);
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);

    const unsigned  int no_of_samples = 1; // Number of samples per light increase this to get better results (but slower)

    // Loop through the lights
    for (auto light: lights) {
        // Loop over number of sample
        for(unsigned int s=0; s < no_of_samples; s++) {
            if (light->sample(hit.position, dir, L_i)) {
                dir_reflected = reflect(-dir, hit.shading_normal);
                radiance += (rho_d * M_1_PIf
                             + rho_s * M_1_PIf * (s + 2) / 2
                               * pow(optix::fmaxf(dot(-r.direction, dir_reflected), 0), s))
                            * L_i * optix::fmaxf(dot(dir, hit.shading_normal), 0);
            }
        }
    }

    return Lambertian::shade(r, hit, emit);
}
