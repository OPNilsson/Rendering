// 02562 Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#ifndef FRESNEL_H
#define FRESNEL_H

inline float fresnel_r_s(float cos_theta1, float cos_theta2, float ior1, float ior2) {
    // Compute the perpendicularly polarized component of the Fresnel reflectance
    float top = ior1 * cos_theta1 - ior2 * cos_theta2;
    float bot = ior1 * cos_theta1 + ior2 * cos_theta2;
    float Rs = top / bot;
    return Rs;
}

inline float fresnel_r_p(float cos_theta1, float cos_theta2, float ior1, float ior2) {
    // Compute the parallelly polarized component of the Fresnel reflectance
    float top = ior2 * cos_theta1 - ior1 * cos_theta2;
    float bot = ior2 * cos_theta1 + ior1 * cos_theta2;
    float Rp = top / bot;
    return Rp;
}

inline float fresnel_R(float cos_theta1, float cos_theta2, float ior1, float ior2) {
    // Compute the Fresnel reflectance using fresnel_r_s(...) and fresnel_r_p(...)
    float r_s = fresnel_r_s(cos_theta1, cos_theta2, ior1, ior2);
    float r_p = fresnel_r_p(cos_theta1, cos_theta2, ior1, ior2);
    float r_s2 = r_s * r_s;
    float r_p2 = r_p * r_p;
    return (r_s2 + r_p2) / 2.0f;
}

#endif // FRESNEL_H
