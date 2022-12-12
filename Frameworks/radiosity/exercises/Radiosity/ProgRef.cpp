#include <math.h>
#include <vector>
#include <float.h>

#include "CGLA/Vec3f.h"
#include "Hemicube.h"
#include "Dataformat.h"
#include "ProgRef.h"

#define M_PI           3.14159265358979323846  /* pi */
#define M_1_PI         0.31830988618379067154  /* 1/pi */

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
	for (unsigned int i = 0; i < polygons.size(); i++)
		polygons[i]->formF = 0.0f;

	// Find the patch with maximum energy
	MyPolygon* maxEnergy = maxenergy_patch();

	// Calculate the form factors between the maximum patch and all other patches.
	// In this function, do it analytically [Watt 2000, Sec. 11.2] or [B, Sec. 31.10].

	// Max Energy Patch variables
	Vec3f n_k = maxEnergy->normal;
	Vec3f C_k = maxEnergy->center;
	float A_k = maxEnergy->area;

	vector<MyPolygon*>::iterator iter;
	for (iter = polygons.begin(); iter != polygons.end(); ++iter)
	{
		if ((*iter) == maxEnergy) {
			continue; // Skip the maximum patch to not check it against itself.
		}

		Vec3f n_j = (*iter)->normal;

		// This gets rid of the green.
		//if (!(dot(n_j, n_k) < 0)) {
		//	continue; // Skip the current patch if it is not visible to the maximum patch.
		//}

		float A_j = (*iter)->area;
		Vec3f C_j = (*iter)->center;

		Vec3f distance = C_j - C_k;
		float r2 = dot(distance, distance);

		Vec3f u_jk = normalize(distance);

		float x = max(dot(n_j, -u_jk), 0.0f);
		float y = max(dot(n_k, u_jk), 0.0f);
		float scale = x * y;

		// Calculate the form factor between the maximum patch and the current patch.
		(*iter)->formF = M_1_PI * (A_j / (r2)) * scale;
	}

	// Return the maximum patch
	return maxEnergy;
}

bool distributeEnergy(MyPolygon* maxP)
{
	if (maxP == nullptr)
		return false;

	// Distribute energy from the maximum patch to all other patches.
	// The energy of the maximum patch is in maxP->unshot_rad (see DataFormat.h).

	vector<MyPolygon*>::iterator iter;
	for (iter = polygons.begin(); iter != polygons.end(); ++iter)
	{
		if ((*iter) == maxP) {
			continue; // Skip the maximum patch to not check it against itself.
		}

		float A_j = (*iter)->area;
		float A_k = maxP->area;

		Vec3f E = maxP->unshot_rad;
		float f_jk = (*iter)->formF;
		Vec3f rho_j = (*iter)->diffuse;

		// Calculate the radiosity of the current patch.
		Vec3f B = E * rho_j * f_jk * (A_k / A_j);

		(*iter)->rad += B;
		(*iter)->unshot_rad += B;
	}

	// Set the unshot radiosity of the maximum patch to zero and return true
	maxP->unshot_rad = Vec3f(0.0, 0.0, 0.0);
	return true;
}

void colorReconstruction()
{
	// Set the colour of all patches by computing their radiances.
	// (Use nodal averaging to compute the colour of all vertices
	//  when you get to this part of the exercises.)

	// Reset all colours to zero
	for (int i = 0; i < vertices.size(); ++i)
	{
		vertices[i]->color = Vec3f(0.0, 0.0, 0.0);
		vertices[i]->colorcount = 0;
	}

	// Sum the colours of all vertices
	vector<MyPolygon*>::iterator iter;
	for (iter = polygons.begin(); iter != polygons.end(); ++iter)
	{
		(*iter)->color = (*iter)->rad * M_1_PI;
		for (int i = 0; i < (*iter)->vertices; ++i)
		{
			vertices[(*iter)->vertex[i]]->color += (*iter)->color;
			vertices[(*iter)->vertex[i]]->colorcount += 1;
		}
	}

	// Average the colours of all vertices
	for (int i = 0; i < vertices.size(); ++i)
	{
		vertices[i]->color /= (float)vertices[i]->colorcount;
	}
}

void renderPatchIDs()
{
	// Render all polygons in the scene as in displayMyPolygons,
	// but set the colour to the patch index using glColor3ub.
	// Look at the Hemicube::getIndex function to see how the
	// indices are read back.
	for (int i = 0; i < polygons.size(); i++) {
		if (4 == polygons[i]->vertices) glBegin(GL_QUADS);
		else if (3 == polygons[i]->vertices) glBegin(GL_TRIANGLES);
		else assert(false); // illegal number of vertices

		//glColor3f(polygons[i]->color[0],polygons[i]->color[1],polygons[i]->color[2]);

		for (int j = 0; j < polygons[i]->vertices; j++) {
			Vec3f position = vertices[polygons[i]->vertex[j]]->position;

			GLubyte i1 = (i + 1) & 255;
			GLubyte i2 = ((i + 1) & (255 << 8)) >> 8;
			GLubyte i3 = ((i + 1) & (255 << 16)) >> 16;

			glColor3ub(i3, i2, i1);

			glVertex3f(position[0], position[1], position[2]);
		}
		glEnd();
	}
}

MyPolygon* calcFF(Hemicube* hemicube)
{
	// Reset all form factors to zero
	for (unsigned int i = 0; i < polygons.size(); i++)
		polygons[i]->formF = 0;

	// Find the patch with maximum energy
	MyPolygon* maxEnergy = maxenergy_patch();

	// Compute a normalized up vector for the maximum patch
	// (use the patch center and one of the patch vertices, for example)
	Vec3f up = normalize(maxEnergy->center - vertices[maxEnergy->vertex[0]]->position);

	// Render patch IDs to the hemicube and read back the index buffer
	hemicube->renderScene(maxEnergy->center, up, maxEnergy->normal, &renderPatchIDs);
	hemicube->readIndexBuffer();

	// Compute form factors by stepping through the pixels of the hemicube
	// and calling hemicube->getDeltaFormFactor(...).
	for (int y = 0; y < hemicube->rendersize; ++y)
		for (int x = 0; x < hemicube->rendersize; ++x)
		{
			int idx = hemicube->getIndex(x, y) - 1;
			if (idx >= 0)
				polygons[idx]->formF += hemicube->getDeltaFormFactor(x, y);
		}

	// Return the maximum patch
	return maxEnergy;
}