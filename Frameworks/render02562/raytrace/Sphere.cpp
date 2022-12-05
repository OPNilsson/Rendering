// 02562 Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <optix_world.h>
#include "HitInfo.h"
#include "Sphere.h"

using namespace optix;

bool Sphere::intersect(const Ray &r, HitInfo &hit, unsigned int prim_idx) const {
    // Implement ray-sphere intersection here.
    //
    // Input:  r                    (the ray to be checked for intersection)
    //         prim_idx             (index of the primitive element in a collection, not used here)
    //
    // Output: hit.has_hit          (set true if the ray intersects the sphere)
    //         hit.dist             (distance from the ray origin to the intersection point)
    //         hit.position         (coordinates of the intersection point)
    //         hit.geometric_normal (the normalized normal of the sphere)
    //         hit.shading_normal   (the normalized normal of the sphere)
    //         hit.material         (pointer to the material of the sphere)
    //        (hit.texcoord)        (texture coordinates of intersection point, not needed for Week 1)
    //
    // Return: True if the ray intersects the sphere, false otherwise
    //
    // Relevant data fields that are available (see Sphere.h and OptiX math library reference)
    // r.origin                     (ray origin)
    // r.direction                  (ray direction)
    // r.tmin                       (minimum intersection distance allowed)
    // r.tmax                       (maximum intersection distance allowed)
    // center                       (sphere center)
    // radius                       (sphere radius)
    // material                     (material of the sphere)
    //
    // Hints: (a) The square root function is called sqrt(x).
    //        (b) There is no need to handle the case where the
    //            discriminant is zero separately.

    // Setup the quadratic equation
    float a = dot(r.direction, r.direction);
    a = 1.0f; // a = 1 according to the lecture slides
    float b = 2.0f * dot((r.origin - center), r.direction);
    float b_2 = b / 2; // b/2
    float c = dot((r.origin - center), (r.origin - center)) - (radius * radius);

    // Calculate the discriminant
    float discriminant = (b_2 * b_2) - c;

    // No intersection if b^2 - c < 0
    if (discriminant < 0.0f) {
        return false;
    }

    // There is no need to handle the case where the discriminant is zero separately but here is the code anyway
    if (discriminant == 0.0f) {
        // If the discriminant is zero the ray grazes the sphere
        // One intersection point
    } else if (discriminant > 0.0f) {
        // If the discriminant is greater than zero the ray intersects the sphere at two points
        // Two intersection points: The entry point and the exit point
    }

    // Calculate distance to the intersection point(s)
    float t_1 = -b_2 - sqrt(b_2 * b_2 - c);
    float t_2 = -b_2 + sqrt(b_2 * b_2 - c);

    bool hit_1 = false;
    bool hit_2 = false;

    // Check if the intersection point(s) are within the ray's tmin and tmax
    if (t_1 >= r.tmin && t_1 <= r.tmax) {
        hit_1 = true;
    }

    if (t_2 >= r.tmin && t_2 <= r.tmax) {
        hit_2 = true;
    }

    if (hit_1 && hit_2) {
        // The closest intersection is the smallest of the two solutions (t_1 and t_2)
        float t = (t_1 < t_2) ? t_1 : t_2;

        hit.has_hit = true;
        hit.dist = t;
        hit.position = r.origin + r.direction * t;
        hit.geometric_normal = normalize(hit.position - center);;
        hit.shading_normal = normalize(hit.position - center);;
        hit.material = &material;

        return true;
    } else if (hit_1) {
        // The closest intersection is t_1
        float t = t_1;

        hit.has_hit = true;
        hit.dist = t;
        hit.position = r.origin + r.direction * t;
        hit.geometric_normal = normalize(hit.position - center);;
        hit.shading_normal = normalize(hit.position - center);;
        hit.material = &material;

        return true;
    } else if (hit_2) {
        // The closest intersection is t_2
        float t = t_2;

        hit.has_hit = true;
        hit.dist = t;
        hit.position = r.origin + r.direction * t;
        hit.geometric_normal = normalize(hit.position - center);;
        hit.shading_normal = normalize(hit.position - center);;
        hit.material = &material;

        return true;
    } else {
        // No intersection
        return false;
    }
}

void Sphere::transform(const Matrix4x4 &m) {
    float3 radius_vec = make_float3(radius, 0.0f, 0.0f) + center;
    radius_vec = make_float3(m * make_float4(radius_vec, 1.0f));
    center = make_float3(m * make_float4(center, 1.0f));
    // The radius is scaled by the X scaling factor.
    // Not ideal, but the best we can do without elipsoids
    radius_vec -= center;
    radius = length(radius_vec);
}

Aabb Sphere::compute_bbox() const {
    Aabb bbox;
    bbox.include(center - make_float3(radius, 0.0f, 0.0f));
    bbox.include(center + make_float3(radius, 0.0f, 0.0f));
    bbox.include(center - make_float3(0.0f, radius, 0.0f));
    bbox.include(center + make_float3(0.0f, radius, 0.0f));
    bbox.include(center - make_float3(0.0f, 0.0f, radius));
    bbox.include(center + make_float3(0.0f, 0.0f, radius));
    return bbox;
}
