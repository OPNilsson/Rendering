// 02562 Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <optix_world.h>
#include "HitInfo.h"
#include "Sphere.h"

using namespace optix;

bool Sphere::intersect(const Ray& r, HitInfo& hit, unsigned int prim_idx) const
{
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
  float b = dot((r.origin - center), r.direction);
  float c = dot((r.origin - center), (r.origin - center)) - (radius * radius);

  // Calculate distance to the intersection point(s)
  float t_1 = -b - sqrt((b * b) - c);
  float t_2 = -b + sqrt((b * b) - c);

  // No intersection if b^2 - c < 0
    if ((b * b) - c < 0.0f){
        return false;
    }

    // Check if the ray is inside the sphere
    if (dot(r.origin - center, r.origin - center) < radius * radius){
        // If the ray is inside the sphere, the intersection point is the ray origin
        hit.has_hit = true;
        hit.dist = 0.0f;
        hit.position = r.origin;
        hit.geometric_normal = normalize(r.origin - center);
        hit.shading_normal = normalize(r.origin - center);
        hit.material = &material;
        return true;
    }

    // Check if the distance to the intersection point is within the ray's tmin and tmax
    if (t_1 >= r.tmin && t_1 <= r.tmax){
        hit.has_hit = true;
        hit.dist = t_1;
        hit.position = r.origin + (t_1 * r.direction);
        hit.geometric_normal = normalize(hit.position - center);
        hit.shading_normal = normalize(hit.position - center);
        hit.material = &material;
        return true;
    }
    else if (t_2 >= r.tmin && t_2 <= r.tmax){
        hit.has_hit = true;
        hit.dist = t_2;
        hit.position = r.origin + (t_2 * r.direction);
        hit.geometric_normal = normalize(hit.position - center);
        hit.shading_normal = normalize(hit.position - center);
        hit.material = &material;
        return true;
    }
    else{
        return false;
    }

    // Check the distant between the ray origin and the sphere center
    float3 distance = r.origin - center;

    // Calculate the discriminant
    float discriminant = dot(distance, r.direction) * dot(distance, r.direction) - dot(distance, distance) + radius * radius;

    // Check if the discriminant is negative
    if (discriminant < 0.0f){
        return false;
    }

    // Calculate the distance to the intersection point
    float distanceToIntersection = -dot(distance, r.direction) - sqrt(discriminant);

    // Check if the distance is negative
    if (distanceToIntersection < 0.0f){
        return false;
    }

    // Calculate the intersection point
    float3 intersectionPoint = r.origin + distanceToIntersection * r.direction;

    // Calculate the normal
    float3 normal = normalize(intersectionPoint - center);

    // Check if the distance is within the ray tmin and tmax
    if (distanceToIntersection < r.tmin || distanceToIntersection > r.tmax){
        return false;
    }

    // Set the hit information
    hit.has_hit = true;
    hit.dist = distanceToIntersection;
    hit.position = intersectionPoint;
    hit.geometric_normal = normal;
    hit.shading_normal = normal;
    hit.material = &material;

    return true;
}

void Sphere::transform(const Matrix4x4& m)
{
  float3 radius_vec = make_float3(radius, 0.0f, 0.0f) + center;
  radius_vec = make_float3(m*make_float4(radius_vec, 1.0f));
  center = make_float3(m*make_float4(center, 1.0f)); 
  // The radius is scaled by the X scaling factor.
  // Not ideal, but the best we can do without elipsoids
  radius_vec -= center;
  radius = length(radius_vec);  
}

Aabb Sphere::compute_bbox() const
{
  Aabb bbox;
  bbox.include(center - make_float3(radius, 0.0f, 0.0f));
  bbox.include(center + make_float3(radius, 0.0f, 0.0f));
  bbox.include(center - make_float3(0.0f, radius, 0.0f));
  bbox.include(center + make_float3(0.0f, radius, 0.0f));
  bbox.include(center - make_float3(0.0f, 0.0f, radius));
  bbox.include(center + make_float3(0.0f, 0.0f, radius));
  return bbox;
}
