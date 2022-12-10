// 02562 Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <optix_world.h>
#include "IndexedFaceSet.h"
#include "ObjMaterial.h"
#include "mt_random.h"
#include "cdf_bsearch.h"
#include "HitInfo.h"
#include "AreaLight.h"

using namespace optix;

bool AreaLight::sample(const float3 &pos, float3 &dir, float3 &L) const {
    const IndexedFaceSet &normals = mesh->normals;
    L = make_float3(0.0f);

    // Compute output and return value given the following information.
    //
    // Input:  pos  (the position of the geometry in the scene)
    //
    // Output: dir  (the direction toward the light)
    //         L    (the radiance received from the direction dir)
    //
    // Return: true if not in shadow
    //
    // Relevant data fields that are available (see Light.h and above):
    // shadows             (on/off flag for shadows)
    // tracer              (pointer to ray tracer)
    // normals             (indexed face set of vertex normals)
    // mesh->face_areas    (array of face areas in the light source)
    //
    // Hints: (a) Use mesh->compute_bbox().center() to get the center of
    //        the light source bounding box.
    //        (b) Use the function get_emission(...) to get the radiance
    //        emitted by a triangle in the mesh.

    // Relevant data formulas using an approximation of the area light:
    // L_r = fr(V/r^2)(w_i dot n) * l_e
    // l_e = sum_index(-w_i dot n_e_index) L_e_index * A_e_index
    // fr(x,w_i, w_o) = dL_r(x, w_o) / dE(x, w_i)  = dL_r(x, w_o) / L_i cos(theta) dw
    // V = 0 || 1 if in shadow || not in shadow
    // r = distance from point to light random light source
    // Monte Carlo estimation of the integral:
    // L_N = L_e + 1/N sum_index(V_index * fr(x, w_j', w) * L_i * cos(theta)/ pdf(wj'))
    // pdf(wj') = cos theta / pi

    // Get random triangle from the light mesh
    double rand_index = mt_random_half_open() * mesh->geometry.no_faces();
    uint3 rand_face = mesh->geometry.face(rand_index);

    double xi_1 = mt_random_half_open();
    double xi_2 = mt_random_half_open();

    float sqrt_xi_1 = sqrt(xi_1);
    float u = 1 - sqrt_xi_1;
    float v = (1 - xi_2) * sqrt_xi_1;
    float w = xi_2 * sqrt_xi_1;

    // Interpolate across the face triangle for smooth shading
    float3 q0 = mesh->geometry.vertex(rand_face.x),
            q1 = mesh->geometry.vertex(rand_face.y),
            q2 = mesh->geometry.vertex(rand_face.z);

    // Setup light ray source
    float3 light_pos = q0 * u + q1 * v + q2 * w;;
    float3 const temp_dir = light_pos - pos;
    float const dir_dot = dot(temp_dir, temp_dir);
    float const distance = sqrt(dir_dot);
    dir = temp_dir / distance;

    // Shadow ray cutoff variables
    float epsilon = 0.0001f; // 10^-4zz
    float t_max = distance - epsilon; // ||p-x|| - epsilon

    // This is also calculating V otherwise known as the visibility
    // If shadows are enabled, check if the point is in shadow
    if (shadows) {
        Ray shadow_ray = Ray(pos, dir, 0.0f, epsilon, t_max);
        HitInfo shadow_hit;
        if (tracer->trace_to_any(shadow_ray, shadow_hit)) {
            return false;
        }
    }

    for (unsigned int i = 0; i < mesh->geometry.no_faces(); i++) {
        L += dot(-dir, normalize(mesh->normals.vertex(i))) / (distance * distance) * get_emission(i) * mesh->face_areas[i];;
    }

    return true;
}

bool AreaLight::emit(Ray &r, HitInfo &hit, float3 &Phi) const {
    // Generate and trace a ray carrying radiance emitted from this area light.
    //
    // Output: r    (the new ray)
    //         hit  (info about the ray-surface intersection)
    //         Phi  (the flux carried by the emitted ray)
    //
    // Return: true if the ray hits anything when traced
    //
    // Relevant data fields that are available (see Light.h and Ray.h):
    // tracer              (pointer to ray tracer)
    // geometry            (indexed face set of triangle vertices)
    // normals             (indexed face set of vertex normals)
    // no_of_faces         (number of faces in triangle mesh for light source)
    // mesh->surface_area  (total surface area of the light source)
    // r.origin            (starting position of ray)
    // r.direction         (direction of ray)

    // Get geometry info
    const IndexedFaceSet &geometry = mesh->geometry;
    const IndexedFaceSet &normals = mesh->normals;
    const float no_of_faces = static_cast<float>(geometry.no_faces());

    // Sample ray origin and direction
    std::cout << "Surface Area: "<< mesh->surface_area << std::endl;
    for(unsigned int i = 0; i <= mesh->surface_area; i++){
        // Pick a random face
        float face = mt_random() * no_of_faces;
    }

    // Trace ray

    // If a surface was hit, compute Phi and return true

    return false;
}

float3 AreaLight::get_emission(unsigned int triangle_id) const {
    const ObjMaterial &mat = mesh->materials[mesh->mat_idx[triangle_id]];
    return make_float3(mat.ambient[0], mat.ambient[1], mat.ambient[2]);
}
