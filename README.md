# Rendering
02562 Rendering Course - Raytracing Engine Implementation    

This course was a development course in where the students where required to develop a raytracing rendering engine. The engine should be able to handle the following techniques that where presented in throughout the course:

- Ray Interceptions with sphere, plane, and triangle objects.
- Lambertian Shading of objects.
- Shadow Casting.
- Reflection and Refraction from mirror and glass objects.
- Phong Illumination.
- Texture Loading.
- Bilinear and nearest neighbor texture filtering.
- Mesh loading with flat shading.
- Smooth shading using interpolated triangle coordinates.
- Axis-Aligned Binary Space Partitioning Tree for mesh subdivision.
- Path traced anti aliasing.
- Progressive sampling of scenes.
- Path traced indirect illumination
- Path traced soft shadows.
- Area, point, directional, and environmental light sources.
- Fresnel Calculated Reflectance.
- Bougouer's Law Absorption.
- Photton Mapping for shading with the use of emitting photons.

A lab journal indicating how each of the above methods were created along side the implementation for each part. The journal goes over my thoughts on implementing each part and is used as a proof of deliverables for each of the course's labs. The journal has screenshots also showing the different techniques explored along mathematical concepts learned in the course. There is also a project journal which is a report over the implementation of the concept that the student was requried to implement outside of the afore mentioned concepts. The entire implementation of this engine has been done individually with only ocational help from TA's and peers in the lectures. 

This course also goes well with the 02561 Computer Graphics course at DTU see Github repo for more.


## Intructions on getting the code to compile and run:
Download the MinGW version of [FreeGLUT](https://www.transmissionzero.co.uk/software/freeglut-devel/) and place it somewhere on the PC this is a required dependency for the code to work. The compiler that is confirmed to build and run the code is CLION. Follow the steps below to run the code:

- Open the FindMyGLUT.cmake file in the cmake folder of the framework. 
- In FIND_PATH, replace `{PROJECT_SOURCE_DIR}/3rdparty/include` with the path to your freeglut (e.g. `C:/programs/freeglut/include`).
- Delete the GL folder from the 3rdparty/include folder of the framework.
- In FIND_LIBRARY, replace `glut32` with `freeglut` and `glut32 glut32d` with `freeglut`. 
- Replace `${PROJECT_SOURCE_DIR}/3rdparty/lib/Debug` and `${PROJECT_SOURCE_DIR}/3rdparty/lib/Release` with the path to the 64-bit binaries in freeglut (e.g. `C:/programs/freeglut/lib/x64`).
- If present, delete the `cmake-build-debug` and/or `cmake-build-release` folder from the CLion project.
- Right-click the project folder and select "Reload CMake Project".
- Copy the 64-bit `freeglut.dll` from the bin folder of freeglut (e.g. `C:/programs/freeglut/bin/x64`) and paste it in the bin folder of the framework.

Now CLION should be able to compile and run the code.
