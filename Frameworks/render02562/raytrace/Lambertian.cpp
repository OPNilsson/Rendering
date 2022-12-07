// 02562 Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <optix_world.h>
#include "HitInfo.h"
#include "Lambertian.h"

using namespace optix;

// The following macro defines 1/PI
#ifndef M_1_PIf
#define M_1_PIf 0.31830988618379067154
#endif

float3 Lambertian::shade(const Ray &r, HitInfo &hit, bool emit) const {
    float3 rho_d = get_diffuse(hit);
    float3 result = make_float3(0.0f);

    // Implement Lambertian reflection here.
    //
    // Input:  r          (the ray that hit the material)
    //         hit        (info about the ray-surface intersection)
    //         emit       (passed on to Emission::shade)
    //
    // Return: radiance reflected to where the ray was coming from
    //
    // Relevant data fields that are available (see Lambertian.h, HitInfo.h, and above):
    // lights             (vector of pointers to the lights in the scene)
    // hit.position       (position where the ray hit the material)
    // hit.shading_normal (surface normal where the ray hit the material)
    // rho_d              (difuse reflectance of the material)
    //
    // Hint: Call the sample function associated with each light in the scene.

    const unsigned  int no_of_samples = 1; // Number of samples per light increase this to get better results (but slower)

    // Loop over all lights
    for (unsigned int i = 0; i < lights.size(); ++i) {
        float3 dir, L;
        // Loop over number of sample
        for(unsigned int s=0; s < no_of_samples; s++){
            // Sample the light
            if (lights[i]->sample(hit.position, dir, L)) {
                // An object was hit by the light ray
                // Compute the cosine of the angle between the light direction and the normal
                float cos_theta = dot(dir, hit.shading_normal);
                if (cos_theta > 0.0f) {
                    result += (rho_d * M_1_PIf) * L * cos_theta;
                }
            }
        }
    }

    return result + Emission::shade(r, hit, emit);
}
