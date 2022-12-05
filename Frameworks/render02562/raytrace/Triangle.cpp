// 02562 Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <optix_world.h>
#include "HitInfo.h"
#include "Triangle.h"

using namespace optix;

bool intersect_triangle(const Ray &ray,
                        const float3 &v0,
                        const float3 &v1,
                        const float3 &v2,
                        float3 &n,
                        float &t,
                        float &beta,
                        float &gamma) {
    // Implement ray-triangle intersection here (see Listing 1 in the lecture note).
    // Note that OptiX also has an implementation, so you can get away
    // with not implementing this function. However, I recommend that
    // you implement it for completeness.

    // Edges of the triangle
    float3 e0 = v1 - v0;
    float3 e1 = v2 - v0;

    // Face normal
    n = cross(e0, e1);

    // Ray-plane intersection
    t = dot((v0 - ray.origin), n) / dot(ray.direction, n);

    if (t >= ray.tmin && t <= ray.tmax) {
        // Decomposition of r(t) - v0
        float3 r_t = ray.origin + t * ray.direction - v0;

        // Barycentric coordinates of r(t)
        float b = (dot((cross(v0 - ray.origin, ray.direction)), e1)) /
                     (dot(ray.direction, n));

        float g = -(dot((cross(v0 - ray.origin, ray.direction)), e0)) /
                      (dot(ray.direction, n));

        float a = 1 - b - g;

        // Make sure we got the correct result
        optix::intersect_triangle(ray, v0, v1, v2, n, t, beta, gamma);
        if (b==beta && g == gamma) {
            //std::cout << "Correct result" << std::endl;
            beta = b;
            gamma = g;
            return true;
        } else {
            //std::cout << "Incorrect result" << std::endl;
            return true;
        }
    }

    return false;
}


bool Triangle::intersect(const Ray &r, HitInfo &hit, unsigned int prim_idx) const {
    // Implement ray-triangle intersection here.
    //
    // Input:  r                    (the ray to be checked for intersection)
    //         prim_idx             (index of the primitive element in a collection, not used here)
    //
    // Output: hit.has_hit          (set true if the ray intersects the triangle)
    //         hit.dist             (distance from the ray origin to the intersection point)
    //         hit.position         (coordinates of the intersection point)
    //         hit.geometric_normal (the normalized normal of the triangle)
    //         hit.shading_normal   (the normalized normal of the triangle)
    //         hit.material         (pointer to the material of the triangle)
    //        (hit.texcoord)        (texture coordinates of intersection point, not needed for Week 1)
    //
    // Return: True if the ray intersects the triangle, false otherwise
    //
    // Relevant data fields that are available (see Triangle.h)
    // r                            (the ray)
    // v0, v1, v2                   (triangle vertices)
    // (t0, t1, t2)                 (texture coordinates for each vertex, not needed for Week 1)
    // material                     (material of the triangle)
    //
    // Hint: Use the function intersect_triangle(...) to get the hit info.
    //       Note that you need to do scope resolution (optix:: or just :: in front
    //       of the function name) to choose between the OptiX implementation and
    //       the function just above this one.

    // Get the triangle intersection variables from the helper function
    float3 n;
    float t;
    float beta;
    float gamma;
    if(::intersect_triangle(r, v0, v1, v2, n, t, beta, gamma)) {
        if (beta >= 0 && gamma >= 0 && beta + gamma <= 1) {
            hit.has_hit = true;
            hit.dist = t;
            hit.position = r.origin + t * r.direction;
            hit.geometric_normal = normalize(n);
            hit.shading_normal = normalize(n);
            hit.material = &material;
            return true;
        }
    }

    return false;
}

void Triangle::transform(const Matrix4x4 &m) {
    v0 = make_float3(m * make_float4(v0, 1.0f));
    v1 = make_float3(m * make_float4(v1, 1.0f));
    v2 = make_float3(m * make_float4(v2, 1.0f));
}

Aabb Triangle::compute_bbox() const {
    Aabb bbox;
    bbox.include(v0);
    bbox.include(v1);
    bbox.include(v2);
    return bbox;
}
