// 02562 Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <optix_world.h>
#include "HitInfo.h"
#include "mt_random.h"
#include "PointLight.h"

using namespace optix;

bool PointLight::sample(const float3 &pos, float3 &dir, float3 &L) const {
    // Compute output and return value given the following information.
    //
    // Input:  pos (the position of the geometry in the scene)
    //
    // Output: dir (the direction toward the light)
    //         L   (the radiance received from the direction dir)
    //
    // Return: true if not in shadow
    //
    // Relevant data fields that are available (see PointLight.h and Light.h):
    // shadows    (on/off flag for shadows)
    // tracer     (pointer to ray tracer)
    // light_pos  (position of the point light)
    // intensity  (intensity of the emitted light)
    //
    // Hint: Construct a shadow ray using the Ray datatype. Trace it using the
    //       pointer to the ray tracer.

    // The distance from the light to the point
    float dist = length(light_pos - pos);

    // The direction from the point to the light
    dir = normalize(light_pos - pos);
    L = intensity / (dist * dist);

    // Shadow ray cutoff variables
    float epsilon = 0.0001f; // 10^-4
    float t_max = dist - epsilon; // ||p-x|| - epsilon

    // If shadows are enabled, check if the point is in shadow
    if (shadows) {
        Ray shadow_ray = Ray(pos, dir, 0.0f, epsilon, t_max);
        HitInfo shadow_hit;
        if (tracer->trace_to_any(shadow_ray, shadow_hit)) {
            return false;
        }
    }

    return true;
}

bool PointLight::emit(Ray &r, HitInfo &hit, float3 &Phi) const {
    // Emit a photon by creating a ray, tracing it, and computing its flux.
    //
    // Output: r    (the photon ray)
    //         hit  (the photon ray hit info)
    //         Phi  (the photon flux)
    //
    // Return: true if the photon hits a surface
    //
    // Relevant data fields that are available (see PointLight.h and Light.h):
    // tracer     (pointer to ray tracer)
    // light_pos  (position of the point light)
    // intensity  (intensity of the emitted light)
    //
    // Hint: When sampling the ray direction, use the function
    //       mt_random() to get a random number in [0,1].

    float3 dir;

    do {
        dir.x = 2.0f * mt_random() - 1.0f;
        dir.y = 2.0f * mt_random() - 1.0f;
        dir.z = 2.0f * mt_random() - 1.0f;
    } while (dot(dir, dir) > 1.0f);
    dir = normalize(dir);

    r = Ray(light_pos, dir, 0, 1e-4, RT_DEFAULT_MAX);
    tracer->trace_to_closest(r, hit);

    if (hit.has_hit) {
        Phi = intensity * 4 * M_PIf;
        return true;
    }


    return false;
}
