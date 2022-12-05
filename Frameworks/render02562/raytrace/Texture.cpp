// 02562 Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <iostream>
#include <optix_world.h>
#include "my_glut.h"
#include "../SOIL/SOIL.h"
#include "Texture.h"

using namespace std;
using namespace optix;

void Texture::load(const char *filename) {
    SOIL_free_image_data(data);
    data = SOIL_load_image(filename, &width, &height, &channels, SOIL_LOAD_AUTO);
    if (!data) {
        cerr << "Error: Could not load texture image file." << endl;
        return;
    }
    int img_size = width * height;
    delete[] fdata;
    fdata = new float4[img_size];
    for (int i = 0; i < img_size; ++i)
        fdata[i] = look_up(i);
    tex_handle = SOIL_create_OGL_texture(data, width, height, channels, tex_handle,
                                         SOIL_FLAG_INVERT_Y | SOIL_FLAG_TEXTURE_REPEATS);
    tex_target = GL_TEXTURE_2D;
}

void Texture::load(GLenum target, GLuint texture) {
    glBindTexture(target, texture);
    glGetTexLevelParameteriv(target, 0, GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(target, 0, GL_TEXTURE_HEIGHT, &height);
    delete[] fdata;
    fdata = new float4[width * height];
    glGetTexImage(target, 0, GL_RGBA, GL_FLOAT, &fdata[0].x);
    tex_handle = texture;
    tex_target = target;
}

float4 Texture::sample_nearest(const float3 &texcoord) const {
    if (!fdata)
        return make_float4(0.0f);

    // Implement texture look-up of nearest texel.
    //
    // Input:  texcoord      (texture coordinates: u = texcoord.x, v = texcoord.y)
    //
    // Return: texel color found at the given texture coordinates
    //
    // Relevant data fields that are available (see Texture.h)
    // fdata                 (flat array of texture data: texel colors in float4 format)
    // width, height         (texture resolution)
    //
    // Hint: Remember to revert the vertical axis when finding the index
    //       into fdata.

    float u = texcoord.x;
    float v = -texcoord.y; // Revert the vertical axis of the texture array

    // The following is done in texture repeat mode

    // Texture coordinates in texture space [0,1] x [0,1]
    float s = u - floor(u);
    float t = v - floor(v);

    // Texture coordinates in image space [0,width] x [0,height]
    float a = s * width;
    float b = t * height;

    // Nearest neighbor filtering [0,1] x [0,1]
    unsigned int U = static_cast<unsigned int>(a + 0.5f) % width;
    unsigned int V = static_cast<unsigned int>(b + 0.5f) % height;

    unsigned int index = U + V * width;

    return fdata[index];
}

float4 Texture::sample_linear(const float3 &texcoord) const {
    if (!fdata)
        return make_float4(0.0f);

    // Implement texture look-up which returns the bilinear interpolation of
    // the four nearest texel neighbors.
    //
    // Input:  texcoord      (texture coordinates: u = texcoord.x, v = texcoord.y)
    //
    // Return: texel color found at the given texture coordinates after
    //         bilinear interpolation
    //
    // Relevant data fields that are available (see Texture.h)
    // fdata                 (flat array of texture data: texel colors in float4 format)
    // width, height         (texture resolution)
    //
    // Hint: Use three lerp operations (or one bilerp) to perform the
    //       bilinear interpolation.

    // Toggle comment the following line to switch to nearest neighbor filtering
    //return sample_nearest(texcoord);

    float u = texcoord.x;
    float v = -texcoord.y; // Revert the vertical axis of the texture array

    // The following is done in texture repeat mode

    // Texture coordinates in texture space [0,1] x [0,1]
    float s = u - floor(u);
    float t = v - floor(v);

    // Texture coordinates in image space [0,width] x [0,height]
    float a = s * width;
    float b = t * height;

    // Bi-linear filtering [0,1] x [0,1]
    unsigned int U = static_cast<unsigned int>(a);
    unsigned int V = static_cast<unsigned int>(b);

    // Weight factors
    float c1 = a - U;
    float c2 = b - V;

    // Get corner texture coordinates
    unsigned int U1 = U % width;
    unsigned int U2 = (U + 1) % width;
    unsigned int V1 = V % height;
    unsigned int V2 = (V + 1) % height;

    // Get corner texture indices
    unsigned int index00 = U1 + V1 * width;
    unsigned int index10 = U2 + V1 * width;
    unsigned int index01 = U1 + V2 * width;
    unsigned int index11 = U2 + V2 * width;

    return lerp(lerp(fdata[index00], fdata[index10], c1), lerp(fdata[index01], fdata[index11], c1), c2);
}

float4 Texture::look_up(unsigned int idx) const {
    idx *= channels;
    switch (channels) {
        case 1: {
            float v = convert(data[idx]);
            return make_float4(v, v, v, 1.0f);
        }
        case 2:
            return make_float4(convert(data[idx]), convert(data[idx]), convert(data[idx]), convert(data[idx + 1]));
        case 3:
            return make_float4(convert(data[idx]), convert(data[idx + 1]), convert(data[idx + 2]), 1.0f);
        case 4:
            return make_float4(convert(data[idx]), convert(data[idx + 1]), convert(data[idx + 2]),
                               convert(data[idx + 3]));
    }
    return make_float4(0.0f);
}

float Texture::convert(unsigned char c) const {
    return (c + 0.5f) / 256.0f;
}
