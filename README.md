# Raytracing / Radiosity Engine
This render engine was development as the main part of a course in where the students where required to develop a raytracing rendering engine. The engine is be able to handle the following techniques that where presented in throughout the course:

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

A lab journal indicating how each of the above methods were created along side the implementation for each part. The journal goes over my thoughts on implementing each part and is used as a proof of the above deliverables. The journal has screenshots also showing the different techniques explored along mathematical concepts learned in the course. There is also a project journal which is a report over the implementation of the radiosity function of the engine. The entire implementation of this engine has been studied for and implemented individually. 

See my WebGL journal for a look into other rendering techniques such as; stencil, bilboards, sciscor planes, and reflection planes.

---

## Intructions on getting the code to compile and run:
Download the MinGW version of [FreeGLUT](https://www.transmissionzero.co.uk/software/freeglut-devel/) and place it somewhere on the PC this is a required dependency for the code to work. The IDE that is confirmed to build and run the code sucessfully are CLION and VSCode. Follow the steps below to run the code:

- Open the FindMyGLUT.cmake file in the cmake folder of the framework. 
- In FIND_PATH, replace `{PROJECT_SOURCE_DIR}/3rdparty/include` with the path to your freeglut (e.g. `C:/programs/freeglut/include`).
- Delete the GL folder from the 3rdparty/include folder of the framework.
- In FIND_LIBRARY, replace `glut32` with `freeglut` and `glut32 glut32d` with `freeglut`. 
- Replace `${PROJECT_SOURCE_DIR}/3rdparty/lib/Debug` and `${PROJECT_SOURCE_DIR}/3rdparty/lib/Release` with the path to the 64-bit binaries in freeglut (e.g. `C:/programs/freeglut/lib/x64`).
- If present, delete the `cmake-build-debug` and/or `cmake-build-release` folder from the CLion project.
- Right-click the project folder and select "Reload CMake Project".
- Copy the 64-bit `freeglut.dll` from the bin folder of freeglut (e.g. `C:/programs/freeglut/bin/x64`) and paste it in the bin folder of the framework.

Now CLION should be able to compile and run the code.

---

## Render Engine Results 
The following are screenshots showing what the render engine is capable of including the rendering of the [Cornell Box](https://www.graphics.cornell.edu/online/box/) raytracing benchmark scene. 

### Cornell Box

#### Path Tracing Result
![cornellBox](https://user-images.githubusercontent.com/45008469/206875979-2c2f5151-9c01-4801-aa6f-66fd0b40a9fc.png)

#### Radiosity Result
![cornellBox_radiosity](https://github.com/OPNilsson/Rendering/blob/1b3a90fa751cf2642b90e7ee676044fb1472a102/Attachments/Radiosity.gif)

### Photon Mapping
![Photton](https://user-images.githubusercontent.com/45008469/206875971-68dad9a3-bbaa-4afd-8d39-5b82b6812406.png)


### Smooth vs Flat Shading of Utah Teapot
|                                                  Flat Shading                                                  |                                                  Smooth Shading                                                  |
| :------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
| ![flat](https://user-images.githubusercontent.com/45008469/206875950-1c6e9b66-35af-4dc0-a2e5-f3c236c0ceb0.png) | ![smooth](https://user-images.githubusercontent.com/45008469/206875958-eb177410-9f9b-4d76-9310-44bc5270ca82.png) |

