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
	//int GlobalW = 256;
	//int GlobalH = 256;
	int GlobalW = 128;
	int GlobalH = 128;


	int pnum;
	vector<pointCoords>coords;

	float SearchRad;

	bool DoSortFrags = false;
	//bool DoSortFrags = true;

	/////////////
	//read file//
	/////////////

	string model_path = "data//";
	string model_name = "bunny2"; SearchRad = 0.00495676;
	//string model_name = "dragon"; SearchRad = 0.00326/2;
		
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
	cout << "----\n";
	cout << "brute force search (ground truth)\n";
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

	}


}











