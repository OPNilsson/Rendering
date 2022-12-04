// 02562 Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <optix_world.h>
#include "HitInfo.h"
#include "Triangle.h"

using namespace optix;

bool intersect_triangle(const Ray& ray, 
                        const float3& v0, 
                        const float3& v1, 
                        const float3& v2, 
                        float3& n,
                        float& t,
                        float& beta,
                        float& gamma)
{
  // Implement ray-triangle intersection here (see Listing 1 in the lecture note).
  // Note that OptiX also has an implementation, so you can get away
  // with not implementing this function. However, I recommend that
  // you implement it for completeness.

    // Find vectors for edges sharing v0
    float3 e0 = v1 - v0;
    float3 e1 = v2 - v0;
    // Find face normal
    float3 normal = cross(e0, e1);
    // Compute ray-plane intersection
    float q = dot(ray.direction, normal);
    if(fabsf(q) < 1.0e-8f) return false;
    q = 1.0f/q;
    float3 o_to_v0 = v0 - ray.origin;
    float distance = dot(o_to_v0, n)*q;
    // Check distance to intersection
    if(distance > ray.tmax || distance < ray.tmin) return false;
    // Find barycentric coordinates
    float3 n_tmp = cross(o_to_v0, ray.direction);
    float y_intersect = dot(n_tmp, e1)*q;
    if(y_intersect < 0.0f) return false;
    float z_intersect = -dot(n_tmp, e0)*q;
    if(z_intersect < 0.0f || y_intersect + z_intersect > 1.0f) return false;
    // Output: n, t, y_intersect, z_intersect
    return true;
}


bool Triangle::intersect(const Ray& r, HitInfo& hit, unsigned int prim_idx) const
{
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

    // Implementation of the Möller-Trumbore algorithm
    const float epsilon = 0.0001f;
    optix::float3 edge1, edge2, h, s, q;
    float  a, f, u, v;
    edge1 = v1 - v0;
    edge2 = v2 - v0;
    h = optix::cross(r.direction, edge2);
    a = optix::dot(edge1, h);
    if (a > -epsilon && a < epsilon)
        return false;    // This ray is parallel to this triangle.
    f = 1.0f / a;
    s = r.origin - v0;
    u = f * optix::dot(s, h);
    if (u < 0.0 || u > 1.0)
        return false;
    q = optix::cross(s, edge1);
    v = f * optix::dot(r.direction, q);
    if (v < 0.0 || u + v > 1.0)
        return false;
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * optix::dot(edge2, q);
    if (t > epsilon) // ray intersection
    {
            hit.has_hit = true;
            hit.dist = t;
            hit.position = r.origin + t * r.direction;
            hit.geometric_normal = optix::normalize(optix::cross(edge1, edge2));
            hit.shading_normal = hit.geometric_normal;
            hit.material = &material;

            if (material.has_texture)
            {
                hit.texcoord = (1 - u - v) * t0 + u * t1 + v * t2;
            }

            return true;
    }
     // This means that there is a line intersection but not a ray intersection.
     return false;
}

void Triangle::transform(const Matrix4x4& m) 
{ 
  v0 = make_float3(m*make_float4(v0, 1.0f)); 
  v1 = make_float3(m*make_float4(v1, 1.0f)); 
  v2 = make_float3(m*make_float4(v2, 1.0f)); 
}

Aabb Triangle::compute_bbox() const
{
  Aabb bbox;
  bbox.include(v0);
  bbox.include(v1);
  bbox.include(v2);
  return bbox;
}
