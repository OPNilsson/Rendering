//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <cuda/whitted.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>

#include <GLFW/glfw3.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm> 
#include <valarray>

#include "QuatTrackBall.h"
#include "ObjScene.h"

//#define USE_IAS // WAR for broken direct intersection of GAS on non-RTX cards

using namespace std;
using namespace sutil;

extern LaunchParams launch_params;

namespace
{
  int32_t width = 1280;
  int32_t height = 720;

  bool resize_dirty = false;
  bool minimized = false;
  bool save_image = false;
  bool export_raw = false;
  bool progressive = true;
  
  // Light state
  bool light_changed = false;
  float theta_i = 54.7356f;
  float phi_i = 45.0f;

  // Mouse state
  bool camera_changed = true;
  QuatTrackBall* trackball = 0;
  int32_t mouse_button = -1;
  float cam_const = 0.41421356f;

  void save_view(const string& filename)
  {
    if(trackball)
    {
      ofstream ofs(filename.c_str(), ofstream::binary);
      if(ofs)
      {
        ofs.write(reinterpret_cast<const char*>(trackball), sizeof(QuatTrackBall));
        ofs.write(reinterpret_cast<const char*>(&cam_const), sizeof(float));
      }
      ofs.close();
      cout << "Camera settings stored in a file called " << filename << endl;
    }
  }

  void load_view(const string& filename)
  {
    if(trackball)
    {
      ifstream ifs_view(filename.c_str(), ifstream::binary);
      if(ifs_view)
      {
        ifs_view.read(reinterpret_cast<char*>(trackball), sizeof(QuatTrackBall));
        ifs_view.read(reinterpret_cast<char*>(&cam_const), sizeof(float));
      }
      ifs_view.close();
      float3 eye, lookat, up;
      float vfov = atanf(cam_const)*360.0f*M_1_PIf;
      trackball->get_view_param(eye, lookat, up);
      cout << "Loaded view: eye [" << eye.x << ", " << eye.y << ", " << eye.z
        << "], lookat [" << lookat.x << ", " << lookat.y << ", " << lookat.z
        << "], up [" << up.x << ", " << up.y << ", " << up.z
        << "], vfov " << vfov << endl;
      camera_changed = true;
    }
  }

  float3 get_light_direction()
  {
    float theta = theta_i*M_PIf/180.0f;
    float phi = phi_i*M_PIf/180.0f;
    float sin_theta = sinf(theta);
    return -make_float3(sin_theta*cosf(phi), sin_theta*sinf(phi), cosf(theta));
  }
}

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
  double xpos, ypos;
  glfwGetCursorPos(window, &xpos, &ypos);

  if(action == GLFW_PRESS)
  {
    mouse_button = button;
    switch(button)
    {
    case GLFW_MOUSE_BUTTON_LEFT:
      trackball->grab_ball(ORBIT_ACTION, make_float2(static_cast<float>(xpos), static_cast<float>(ypos)));
      break;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      trackball->grab_ball(DOLLY_ACTION, make_float2(static_cast<float>(xpos), static_cast<float>(ypos)));
      break;
    case GLFW_MOUSE_BUTTON_RIGHT:
      trackball->grab_ball(PAN_ACTION, make_float2(static_cast<float>(xpos), static_cast<float>(ypos)));
      break;
    }
  }
  else
  {
    trackball->release_ball();
    mouse_button = -1;
  }
}

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
  if(mouse_button >= 0)
  {
    trackball->roll_ball(make_float2(static_cast<float>(xpos), static_cast<float>(ypos)));
    camera_changed = true;
  }
}

static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
  // Keep rendering at the current resolution when the window is minimized.
  if(minimized)
    return;

  // Output dimensions must be at least 1 in both x and y.
  ensureMinimumSize(res_x, res_y);

  width = res_x;
  height = res_y;
  camera_changed = true;
  resize_dirty = true;
}

static void windowIconifyCallback(GLFWwindow* window, int32_t iconified)
{
  minimized = (iconified > 0);
}

static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
  if(action == GLFW_PRESS || action == GLFW_REPEAT)
  {
    switch(key)
    {
    case GLFW_KEY_Q:      // Quit the program using <Q>
    case GLFW_KEY_ESCAPE: // Quit the program using <esc>
      glfwSetWindowShouldClose(window, true);
      break;
    case GLFW_KEY_S:      // Save the rendered image using <S>
      if(action == GLFW_PRESS)
        save_image = true;
      break;
    case GLFW_KEY_R:      // Toggle progressive rendering using <R>
      if(action == GLFW_PRESS)
      {
        cout << "Samples per pixel: " << launch_params.subframe_index << endl;
        progressive = !progressive;
        cout << "Progressive sampling is " << (progressive ? "on." : "off.") << endl;
      }
      break;
    case GLFW_KEY_Z:      // Zoom using 'z' or 'Z'
    {
      int rshift = glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT);
      int lshift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT);
      cam_const *= rshift || lshift ? 1.05f : 1.0f/1.05f;
      camera_changed = true;
      cout << "Vertical field of view: " << atanf(cam_const)*360.0f*M_1_PIf << endl;
      break;
    }
    case GLFW_KEY_O:      // Save current view to a file called view using <O> (output)
      if(action == GLFW_PRESS)
        save_view("view");
      break;
    case GLFW_KEY_I:      // Load current view from a file called view using <I> (input)
      if(action == GLFW_PRESS)
        load_view("view");
      break;
    case GLFW_KEY_E:      // Export raw image data using 'e'
      {
        if(action == GLFW_PRESS)
        { 
          save_image = true;
          export_raw = true;
        }
        break;
      }
    case GLFW_KEY_KP_ADD: // Increment the angle of incidence using '+'
      {
        theta_i = fminf(theta_i + 1.0f, 90.0f);
        cout << "Angle of incidence: " << static_cast<int>(theta_i) << endl;
        light_changed = true;
        break;
      }
    case GLFW_KEY_KP_SUBTRACT: // Decrement the angle of incidence using '-'
      {
        theta_i = fmaxf(theta_i - 1.0f, -90.0f);
        cout << "Angle of incidence: " << static_cast<int>(theta_i) << endl;
        light_changed = true;
        break;
      }
    }
  }
}

static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
  //if(trackball.wheelEvent((int)yscroll))
  //  camera_changed = true;
}

//------------------------------------------------------------------------------
//
// Helper functions
//
//------------------------------------------------------------------------------

void printUsageAndExit(const char* argv0)
{
  cerr << "Usage  : " << argv0 << " [options] any_object.obj [another.obj ...]" << endl
    << "Options: --help    | -h                 Print this usage message" << endl
    << "         --shader  | -sh <shader>       Specify the closest hit program to be used for shading" << endl
    << "                   | -env <filename>    Specify the environment map to be loaded in panoramic format" << endl
    << "                   | -bgc <r> <g> <b>   Specify RGB background color (not used if env is available)" << endl
    << "         --dim=<width>x<height>         Set image dimensions; defaults to 768x768" << endl
    << "         --no-gl-interop                Disable GL interop for display" << endl
    << "         --file    | -f <filename>      File for image output" << endl
    << "         --samples | -s                 Number of samples per pixel if rendering to file (default 16)" << endl
    << "                   | -dir <th> <ph>     Direction of default light in spherical coordinates (polar <th>, azimuth <ph>)" << endl
    << "                   | -rad <r> <g> <b>   Specify RGB radiance of default directional light (default PI)";
  exit(0);
}

// Avoiding case sensitivity
void lower_case(char& x)
{
  x = tolower(x);
}
inline void lower_case_string(std::string& s)
{
  for_each(s.begin(), s.end(), lower_case);
}

void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer, ObjScene& scene)
{
  bool reset = camera_changed || light_changed || resize_dirty;

  if(resize_dirty)
  {
    trackball->set_screen_window(width, height);
    scene.handleResize(output_buffer, width, height);
    resize_dirty = false;
  }
  if(camera_changed)
  {
    float3 eye, lookat, up;
    trackball->get_view_param(eye, lookat, up);
    scene.camera.setEye(eye);
    scene.camera.setLookat(lookat);
    scene.camera.setUp(up);
    scene.camera.setFovY(atanf(cam_const)*360.0f*M_1_PIf);
    scene.handleCameraUpdate();
    camera_changed = false;
  }
  if(light_changed)
  {
    scene.light_dir = get_light_direction();
    scene.handleLightUpdate();
    light_changed = false;
  }

  // Update params on device
  if(reset)
  {
    launch_params.subframe_index = 0;
    int size_buffer = width*height*4;
    uchar4* result_buffer_data = output_buffer.map();
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(result_buffer_data), 0, size_buffer*sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(launch_params.accum_buffer), 0, size_buffer*sizeof(float)));
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
  }
}

void displaySubframe(
        sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
        sutil::GLDisplay&                 gl_display,
        GLFWwindow*                       window )
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
            output_buffer.width(),
            output_buffer.height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer.getPBO());
}

void saveImage(sutil::CUDAOutputBuffer<uchar4>& output_buffer, string outfile)
{
  sutil::ImageBuffer buffer;
  buffer.data = output_buffer.getHostPointer();
  buffer.width = output_buffer.width();
  buffer.height = output_buffer.height();
  buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
  sutil::saveImage(outfile.c_str(), buffer, true);
  cout << "Rendered image stored in " << outfile << endl;
}

void exportRawImage(string outfile)
{
  // Get image info
  size_t name_end = outfile.find_last_of('.');
  string name = outfile.substr(0, name_end);
  unsigned int frame = launch_params.subframe_index + 1;

  // Write image info in .txt-file 
  ofstream ofs_data(name + ".txt");
  if(ofs_data.bad())
    return;
  ofs_data << frame << endl << width << " " << height << endl;
  ofs_data << theta_i << " " << phi_i;
  ofs_data.close();

  // Copy buffer data from device to host
  int size_buffer = width*height*4;
  float* mapped = new float[size_buffer];
  CUDA_CHECK(cudaMemcpyAsync(mapped, launch_params.accum_buffer, size_buffer*sizeof(float), cudaMemcpyDeviceToHost, 0));

  // Export image data to binary .raw-file
  ofstream ofs_image;
  ofs_image.open(name + ".raw", ios::binary);
  if(ofs_image.bad())
  {
    cerr << "Error when exporting file" << endl;
    return;
  }

  int size_image = width*height*3;
  float* converted = new float[size_image];
  float average = 0.0f;
  for(int i = 0; i < size_image/3; ++i)
  {
    for(int j = 0; j < 3; ++j)
    {
      float value = mapped[i*4 + j];
      converted[i*3 + j] = value;
      average += value;
    }
  }
  average /= static_cast<float>(size_image);
  delete[] mapped;
  ofs_image.write(reinterpret_cast<const char*>(converted), size_image*sizeof(float));
  ofs_image.close();
  delete[] converted;
  cout << "Exported buffer to " << name << ".raw (avg: " << average << ")" << endl;
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
  sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

  //
  // Parse command line options
  //
  vector<string> filenames;
  string filename;
  string shadername;
  string envname;
  string outfile;
  bool outfile_selected = false;
  float3 light_dir = make_float3(-1.0f);
  float3 emission = make_float3(M_PIf);
  unsigned int samples = 16;
  launch_params.miss_color = make_float3(0.8f, 0.9f, 1.0f);

  for( int i = 1; i < argc; ++i )
  {
    const std::string arg = argv[i];
    if(arg == "--help" || arg == "-h")
    {
      printUsageAndExit( argv[0] );
    }
    else if(arg == "-sh" || arg == "--shader")
    {
      if(i == argc - 1)
        printUsageAndExit(argv[0]);
      shadername = argv[++i];
      lower_case_string(shadername);
    }
    else if(arg == "-env")
    {
      if(i == argc - 1)
        printUsageAndExit(argv[0]);

      envname = argv[++i];
      string file_extension;
      size_t idx = envname.find_last_of('.');
      if(idx < envname.length())
      {
        file_extension = envname.substr(idx, envname.length() - idx);
        lower_case_string(file_extension);
      }
      if(file_extension == ".png" || file_extension == ".ppm" || file_extension == ".hdr")
        lower_case_string(envname);
      else
      {
        cerr << "Please use environment maps in .png or .ppm  or .hdr format. Received: '" << envname << "'" << endl;
        printUsageAndExit(argv[0]);
      }
    }
    else if(arg == "-bgc")
    {
      if(i >= argc - 3)
        printUsageAndExit(argv[0]);
      launch_params.miss_color.x = static_cast<float>(atof(argv[++i]));
      launch_params.miss_color.y = static_cast<float>(atof(argv[++i]));
      launch_params.miss_color.z = static_cast<float>(atof(argv[++i]));
    }
    else if(arg.substr(0, 6) == "--dim=")
    {
      const std::string dims_arg = arg.substr(6);
      sutil::parseDimensions(dims_arg.c_str(), width, height);
    }
    else if(arg == "--no-gl-interop")
    {
      output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
    }
    else if(arg == "--file" || arg == "-f")
    {
      if(i >= argc - 1)
        printUsageAndExit(argv[0]);
      outfile = argv[++i];
      outfile_selected = true;
    }
    else if(arg == "--samples" || arg == "-s")
    {
      if(i >= argc - 1)
        printUsageAndExit(argv[0]);
      samples = atoi(argv[++i]);
    }
    else if(arg == "-dir")
    {
      if(i >= argc - 2)
        printUsageAndExit(argv[0]);
      theta_i = static_cast<float>(atof(argv[++i]));
      phi_i = static_cast<float>(atof(argv[++i]));
      light_dir = get_light_direction();
    }
    else if(arg == "-rad")
    {
      if(i >= argc - 3)
        printUsageAndExit(argv[0]);
      emission.x = static_cast<float>(atof(argv[++i]));
      emission.y = static_cast<float>(atof(argv[++i]));
      emission.z = static_cast<float>(atof(argv[++i]));
    }
    else
    {
      filename = argv[i];
      string file_extension;
      size_t idx = filename.find_last_of('.');
      if(idx < filename.length())
      {
        file_extension = filename.substr(idx, filename.length() - idx);
        lower_case_string(file_extension);
      }
      if(file_extension == ".obj")
      {
        filenames.push_back(filename);
        lower_case_string(filenames.back());
      }
      else
      {
        cerr << "Unknown option or not an obj file: '" << arg << "'" << endl;
        printUsageAndExit(argv[0]);
      }
    }
  }
  if(filenames.size() == 0)
    filenames.push_back(string(SAMPLES_DIR) + "/models/cow_vn.obj");
  if(!outfile_selected)
  {
    size_t name_start = filenames.back().find_last_of("\\/") + 1;
    size_t name_end = filenames.back().find_last_of('.');
    outfile = filenames.back().substr(name_start < name_end ? name_start : 0, name_end - name_start) + ".png";
  }

  try
  {
    ObjScene scene(filenames, shadername, envname, width, height, light_dir, emission);
    scene.initScene();

    camera_changed = true;
    trackball = new QuatTrackBall(scene.camera.lookat(), length(scene.camera.lookat() - scene.camera.eye()), width, height);

    if(!outfile_selected)
    {
      GLFWwindow* window = sutil::initUI( "render_OptiX", scene.width, scene.height );
      glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
      glfwSetCursorPosCallback    ( window, cursorPosCallback     );
      glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
      glfwSetWindowIconifyCallback( window, windowIconifyCallback );
      glfwSetKeyCallback          ( window, keyCallback           );
      glfwSetScrollCallback       ( window, scrollCallback        );
      glfwSetWindowUserPointer    ( window, &launch_params         );

      //
      // Render loop
      //
      {
        sutil::CUDAOutputBuffer<uchar4> output_buffer(output_buffer_type, scene.width, scene.height);
        sutil::GLDisplay gl_display;

        std::chrono::duration<double> state_update_time(0.0);
        std::chrono::duration<double> render_time(0.0);
        std::chrono::duration<double> display_time(0.0);

        do
        {
          auto t0 = std::chrono::steady_clock::now();
          glfwPollEvents();

          updateState(output_buffer, scene);
          auto t1 = std::chrono::steady_clock::now();
          state_update_time += t1 - t0;
          t0 = t1;

          if(progressive || launch_params.subframe_index == 0)
            scene.launchSubframe(output_buffer);
          t1 = std::chrono::steady_clock::now();
          render_time += t1 - t0;
          t0 = t1;

          displaySubframe(output_buffer, gl_display, window);
          t1 = std::chrono::steady_clock::now();
          display_time += t1 - t0;

          if(progressive)
            sutil::displayStats( state_update_time, render_time, display_time );

          glfwSwapBuffers(window);

          if(save_image)
          {
            if(export_raw)
            {
              exportRawImage(outfile);
              export_raw = false;
            }
            else
              saveImage(output_buffer, outfile);
            save_image = false;
          }
          if(progressive)
            ++launch_params.subframe_index;
        }
        while( !glfwWindowShouldClose( window ) );
        CUDA_SYNC_CHECK();
      }

      sutil::cleanupUI( window );
    }
    else
    {
		  if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
		  {
			  sutil::initGLFW(); // For GL context
			  sutil::initGL();
		  }

		  sutil::CUDAOutputBuffer<uchar4> output_buffer(output_buffer_type, scene.width, scene.height);
      updateState(output_buffer, scene);

      cout << "Rendering";
      unsigned int dot = max(20u, samples/20u);
      chrono::duration<double> render_time(0.0);
      auto t0 = chrono::steady_clock::now();
      for(unsigned int i = 0; i < samples; ++i)
      {
        scene.launchSubframe(output_buffer);
        ++launch_params.subframe_index;
        if((i + 1)%dot == 0) cerr << ".";
      }
      auto t1 = chrono::steady_clock::now();
      render_time = t1 - t0;
      cout << endl << "Time: " << render_time.count() << endl;

      exportRawImage(outfile);
      saveImage(output_buffer, outfile);
      if(output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
      {
        output_buffer.deletePBO();
        glfwTerminate();
      }
    }
    delete trackball;
  }
  catch( std::exception& e )
  {
      std::cerr << "Caught exception: " << e.what() << "\n";
      return 1;
  }
  return 0;
}
