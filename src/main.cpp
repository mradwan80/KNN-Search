#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <stack>
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include <set>
#include "DDS.h"
using namespace std;

void main()
{
	//int GlobalW = 4096;
	//int GlobalH = 4096;
	//int GlobalW = 1024;
	//int GlobalH = 1024;
	//int GlobalW = 512;
	//int GlobalH = 512;
	//int GlobalW = 256;
	//int GlobalH = 256;
	int GlobalW = 128;
	int GlobalH = 128;
	//int GlobalW = 64;
	//int GlobalH = 64;
	//int GlobalW = 32;
	//int GlobalH = 32;
	//int GlobalW = 16;
	//int GlobalH = 16;
	//int GlobalW = 8;
	//int GlobalH = 8;
	//int GlobalW = 4;
	//int GlobalH = 4;
	//int GlobalW = 2;
	//int GlobalH = 2;
	//int GlobalW = 1600;
	//int GlobalH = 1200;
	//int GlobalW = 800;
	//int GlobalH = 600;


	/*glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);

	glutInitWindowSize(GlobalW, GlobalH);
	glutInitWindowPosition(0, 0);
	glutCreateWindow(argv[0]);

	glewExperimental = GL_TRUE; // for experimental extensions
	GLenum err = glewInit(); // initialize GLEW

	glEnable(GL_PROGRAM_POINT_SIZE);*/

	int pnum;
	vector<pointCoords>coords;

	float SearchRad;

	/////////////
	//read file//
	/////////////

	string model_path = "C://models//OFF//";
	//string model_name = "bunny2"; SearchRad = 0.00495676;
	//string model_name = "armadillo";  SearchRad = 0.904385;
	//string model_name = "ant"; SearchRad = 0.0278831;
	//string model_name = "bird"; SearchRad = 0.028972;
	//string model_name = "glass"; SearchRad = 0.0230571;
	string model_name = "cup"; SearchRad = 0.0264987;
	//string model_name = "human"; SearchRad = 0.024626;
	//string model_name = "elephant"; SearchRad = 0.0132349;
	//string model_name = "dragon3";
	//string model_name = "dragon2";
	
	
	ifstream input_file(model_path + model_name + ".OFF");


	int tnum;
	int dummyi;
	char ch;
	input_file >> ch >> ch >> ch;
	input_file >> pnum >> tnum >> dummyi;
	coords.resize(pnum);
	vector<int>connectivity(3 * tnum);
	for (int i = 0; i < pnum; i++)
	{
		input_file >> coords[i].x >> coords[i].y >> coords[i].z;
	}
	for (int i = 0; i < tnum; i++)
	{
		input_file >> dummyi >> connectivity[3 * i] >> connectivity[3 * i + 1] >> connectivity[3 * i + 2];
	}
	input_file.close();

	cout << "points number: " << pnum << "\n";

	///////////////
	//set matrices//
	///////////////

	float minx = std::numeric_limits<float>::max();
	float miny = std::numeric_limits<float>::max();
	float minz = std::numeric_limits<float>::max();
	float maxx = std::numeric_limits<float>::lowest();
	float maxy = std::numeric_limits<float>::lowest();
	float maxz = std::numeric_limits<float>::lowest();
	for (int i = 0; i < pnum; i++)
	{
		if (minx > coords[i].x)
			minx = coords[i].x;
		if (miny > coords[i].y)
			miny = coords[i].y;
		if (minz > coords[i].z)
			minz = coords[i].z;

		if (maxx < coords[i].x)
			maxx = coords[i].x;
		if (maxy < coords[i].y)
			maxy = coords[i].y;
		if (maxz < coords[i].z)
			maxz = coords[i].z;
	}
	float midx = (minx + maxx) / 2;
	float midy = (miny + maxy) / 2;
	float midz = (minz + maxz) / 2;

	//special settings, for building the DDS ON the mesh !!!//
	glm::mat4 ViewMat = glm::translate(glm::vec3(0, 0, 0));
	glm::mat4 vmMat = ViewMat; //no model matrix//
	float left, right, bottom, top;
	float ViewWidth;
	if ((maxx - minx) > (maxy - miny))
		ViewWidth = 1.02 * (maxx - minx);
	else
		ViewWidth = 1.02 * (maxy - miny);

	left = midx - 0.5 * ViewWidth;
	right = midx + 0.5 * ViewWidth;
	bottom = midy - 0.5 * ViewWidth;
	top = midy + 0.5 * ViewWidth;

	float Near = minz;
	float Far = maxz;
	glm::mat4 ProjectionOrthoMat = glm::ortho(left, right, bottom, top, Near, Far);
	glm::mat4 pvmOrthoMat = ProjectionOrthoMat;

	///////////////
	//compute dds//
	///////////////

	//bool DoSortFrags = false;
	bool DoSortFrags = true;

	
	DDS* dds = new DDS(GlobalW, GlobalH, ViewWidth, &coords, vmMat, pvmOrthoMat);
	dds->BuildDDS(DoSortFrags);
	cout << "dds finished\n";
	//delete dds;

	
	//SearchRad = ViewWidth / GlobalW * 2;
	cout << "search rad: " << SearchRad << "\n";

	int k = 7;
	dds->findKNN(k, SearchRad, DoSortFrags);
	cout << "finding neighbors finished\n";


	//for testing purposes
	float maxf = numeric_limits<float>::max();
	for (int i = 0; i < 10; i++)
	{
		vector<float>mindistance(k, maxf);
		vector<int>minvertex(k);

		for (int j = 0; j < coords.size(); j++)
		{
			float dist = (coords[i].x - coords[j].x) * (coords[i].x - coords[j].x) + (coords[i].y - coords[j].y) * (coords[i].y - coords[j].y) + (coords[i].z - coords[j].z) * (coords[i].z - coords[j].z);

			bool fin = false;
			for (int u = 0; u < k && !fin; u++)
			{
				if (dist < mindistance[u])
				{

					for (int o = k - 1; o > u; o--)
					{
						mindistance[o] = mindistance[o - 1];
						minvertex[o] = minvertex[o - 1];
					}
					mindistance[u] = dist;
					minvertex[u] = j;
					fin = true;
				}
			}

		}

		cout << "k neareset neighbors of " << i << " are: ";
		for (int u = 0; u < k; u++)
		{
			if (sqrt(mindistance[u]) <= SearchRad)
				cout << minvertex[u] << " ";
		}
		cout << "\n";

		/*cout << "k neareset neighbors of " << i << " are: ";
		for (int u = 0; u < k; u++)
		{
			if (sqrt(mindistance[u]) <= SearchRad)
				std::cout << "[" << coords[minvertex[u]].x << ", " << coords[minvertex[u]].y << ", " << coords[minvertex[u]].z << "], ";
		}
		cout << "\n";*/
	}


}











