// 02562 Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <iostream>
#include <optix_world.h>
#include "mt_random.h"
#include "Shader.h"
#include "HitInfo.h"
#include "RayCaster.h"

using namespace std;
using namespace optix;

float3 RayCaster::compute_pixel(unsigned int x, unsigned int y) const {
    // Use the scene and its camera
    // to cast a ray that computes the color of the pixel at index (x, y).
    //
    // Input:  x, y        (pixel index)
    //
    // Return: Result of tracing a ray through the pixel at index (x, y).
    //
    // Relevant data fields that are available (see RayCaster.h and Scene.h)
    // win_to_ip           (pixel size (width, height) in the image plane)
    // lower_left          (lower left corner of the film in the image plane)
    // scene               (scene with access to the closest_hit function)
    // scene->get_camera() (camera in the scene)
    //
    // Hints: (a) Use the function get_shader(...) to get the shader of the
    //            intersected material after the ray has been traced.
    //        (b) Use get_background(...) if the ray does not hit anything.

    float3 result;
    float2 pos = make_float2(x, y) * win_to_ip + lower_left;

    // Start of the Antialiasing implementation (Task 3)
    // This implementation is based on the code from the supplementary material
    for(unsigned int i = 0; i < subdivs; i++) {
        for (int j = 0; j < subdivs; ++j) {
            float2 jitter_pos = jitter[i * subdivs + j];
            Ray ray = scene->get_camera()->get_ray(pos + jitter_pos);
            HitInfo hit_info;
            if (scene->closest_hit(ray, hit_info)) {
                result += scene->get_shader(hit_info)->shade(ray, hit_info);
            } else {
                result += get_background(ray.direction);
            }
        }
    }

    // Sub-pixels = subdivs * subdivs (subdivs = 1 for no antialiasing)
    return result / (subdivs * subdivs);
}

float3 RayCaster::get_background(const float3 &dir) const {
    if (!sphere_tex)
        return background;
    return make_float3(sphere_tex->sample_linear(dir));
}

void RayCaster::increment_pixel_subdivs() {
    ++subdivs;
    compute_jitters();
    cout << "Rays per pixel: " << subdivs * subdivs << endl;
}

void RayCaster::decrement_pixel_subdivs() {
    if (subdivs > 1) {
        --subdivs;
        compute_jitters();
    }
    cout << "Rays per pixel: " << subdivs * subdivs << endl;
}

void RayCaster::compute_jitters() {
    float aspect = width / static_cast<float>(height);
    win_to_ip.x = win_to_ip.y = 1.0f / static_cast<float>(height);
    lower_left = (win_to_ip - make_float2(aspect, 1.0f)) * 0.5f;
    step = win_to_ip / static_cast<float>(subdivs);

//    cout << "Pixel size: " << win_to_ip.x << " " << win_to_ip.y << endl;
//    cout << "Lower left: " << lower_left.x << " " << lower_left.y << endl;
//    cout << "Step: " << step.x << " " << step.y << endl;
//    cout<< "Subdivs: " << subdivs << endl;

    jitter.resize(subdivs * subdivs);
    for (unsigned int i = 0; i < subdivs; ++i)
        for (unsigned int j = 0; j < subdivs; ++j)
            jitter[i * subdivs + j] = make_float2(safe_mt_random() + j, safe_mt_random() + i) * step - win_to_ip * 0.5f;

//    cout << "Jitter: " << jitter[0].x << " " << jitter[0].y << endl;
}
