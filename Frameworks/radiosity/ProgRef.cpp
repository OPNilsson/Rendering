#include <math.h>
#include <vector>
#include <float.h>

#include "CGLA/Vec3f.h"
#include "Hemicube.h"
#include "Dataformat.h"
#include "ProgRef.h"

using namespace CGLA;
using namespace std;

extern vector<MyPolygon*> polygons;
extern vector<MyVertex*> vertices;

bool polygon_cmp(MyPolygon* mp1, MyPolygon* mp2)
{ 
  return sqr_length(mp1->unshot_rad) < sqr_length(mp2->unshot_rad);
}

MyPolygon* maxenergy_patch()
{
  return *std::max_element(polygons.begin(), polygons.end(), polygon_cmp);
}

MyPolygon* calcAnalyticalFF()
{
  // Reset all form factors to zero
  for(unsigned int i = 0; i < polygons.size(); i++)
    polygons[i]->formF = 0.0f;

  // Find the patch with maximum energy

  // Calculate the form factors between the maximum patch and all other patches.
  // In this function, do it analytically [Watt 2000, Sec. 11.2] or [B, Sec. 31.10].

  // Return the maximum patch
  return 0;
}

bool distributeEnergy(MyPolygon* maxP)
{
  if(maxP == 0)
  	return false;

  // Distribute energy from the maximum patch to all other patches.
  // The energy of the maximum patch is in maxP->unshot_rad (see DataFormat.h).

  // Set the unshot radiosity of the maximum patch to zero and return true
  return true;
}

void colorReconstruction()
{
  // Set the colour of all patches by computing their radiances. 
  // (Use nodal averaging to compute the colour of all vertices 
  //  when you get to this part of the exercises.)

}


void renderPatchIDs()
{
  // Render all polygons in the scene as in displayMyPolygons,
  // but set the colour to the patch index using glColor3ub.
  // Look at the Hemicube::getIndex function to see how the
  // indices are read back.

}

MyPolygon* calcFF(Hemicube* hemicube)
{
  // Reset all form factors to zero
  for(unsigned int i = 0; i < polygons.size(); i++)
  	polygons[i]->formF = 0; 
  
  // Find the patch with maximum energy

  // Compute a normalized up vector for the maximum patch 
  // (use the patch center and one of the patch vertices, for example)

  // Render patch IDs to the hemicube and read back the index buffer

  // Compute form factors by stepping through the pixels of the hemicube
  // and calling hemicube->getDeltaFormFactor(...).
    
  // Return the maximum patch
  return 0;
}
