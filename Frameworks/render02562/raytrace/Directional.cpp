// 02562 Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <sstream>
#include <string>
#include <optix_world.h>
#include "HitInfo.h"
#include "Directional.h"

using namespace std;
using namespace optix;

bool Directional::sample(const float3 &pos, float3 &dir, float3 &L) const {
    // Compute output and return value given the following information.
    //
    // Input:  pos (the position of the geometry in the scene)
    //
    // Output: dir (the direction toward the light)
    //         L   (the radiance received from the direction dir)
    //
    // Return: true if not in shadow
    //
    // Relevant data fields that are available (see Directional.h and Light.h):
    // shadows    (on/off flag for shadows)
    // tracer     (pointer to ray tracer)
    // light_dir  (direction of the emitted light)
    // emission   (radiance of the emitted light)

    // The direction toward the light is opposite to the light direction
    dir = -light_dir;

    // Shadow ray cutoff variables
    float epsilon = 0.0001f; // 10^-4
    float t_max = RT_DEFAULT_MAX - epsilon; // cannot be calculated because of the infinite light source

    // If shadows are enabled, check if the point is in shadow
    if (shadows) {
        Ray shadow_ray = Ray(pos, dir, 0.0f, epsilon, t_max);
        HitInfo shadow_hit;

        if (tracer->trace_to_any(shadow_ray, shadow_hit)) {
            return false;
        }
    }

    L = emission;

    return true;
}

string Directional::describe() const {
    ostringstream ostr;
    ostr << "Directional light (emitted radiance " << emission << ", direction " << light_dir << ").";
    return ostr.str();
}
