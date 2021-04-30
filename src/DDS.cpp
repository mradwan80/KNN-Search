#include "DDS.h"
#include <chrono>
#include <iostream>
//#include "OpenGlClasses.h"
#include <glm/gtc/type_ptr.hpp>
#include<thrust/device_ptr.h>
#include<thrust/sort.h>
#include <thrust/execution_policy.h>
//#include "ClassesDefinitions.h"
using namespace std;

DDS::DDS() {}

DDS::DDS(int w, int h, float viewWidthI, vector<pointCoords>* Pos, glm::mat4 vmMatI, glm::mat4 pvmMatI)
{
	globalW = w;
	globalH = h;
	viewWidth = viewWidthI;

	vxPos = Pos;

	vmMat = vmMatI;
	pvmMat = pvmMatI;

	//glViewport(0, 0, w, h);

	//create vxIndex//
	vxIndex.reserve(vxPos->size());
	for (int i = 0; i < vxPos->size(); i++)
		vxIndex.push_back(i);

	//SetLighting(); //no colors for now

}

void DDS::PrepareInput()
{
	float milliseconds;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	//allocate
	cudaMalloc((void**)&matrixVM, 16 * sizeof(float));
	cudaMalloc((void**)&matrixPVM, 16 * sizeof(float));
	cudaMalloc((void**)&vpos, vxPos->size() * 3 * sizeof(float));
	cudaMalloc((void**)&xfcount, globalW * globalH * sizeof(int));
	cudaMalloc((void**)&xfoffset, globalW * globalH * sizeof(int));

	//copy
	cudaMemcpy(matrixVM, (float*)glm::value_ptr(vmMat), 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(matrixPVM, (float*)glm::value_ptr(pvmMat), 16 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(vpos, vxPos->data(), vxPos->size() * 3 * sizeof(float), cudaMemcpyHostToDevice);

	FillAllWithValue(xfcount, globalW * globalH, 0);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	if(debug_details)
		cout << "***Preparation time: " << milliseconds << '\n';


}

void DDS::CountFrags()
{
	float milliseconds;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	CountFragsCuda(vxPos->size(), globalW, globalH, viewWidth, matrixVM, matrixPVM, vpos, xfcount);


	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	if (debug_details)
		cout << "***CountFrags time: " << milliseconds << '\n';


}

void DDS::TestCountFrags()
{
	int MoreThanOne = 0;

	//get vector back
	int* xfcountHost = new int[globalW * globalH];
	cudaMemcpy(xfcountHost, xfcount, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);

	//make sure all either 1s or 0s
	int ctr = 0;
	int occPixels = 0;
	for (int i = 0; i < globalW * globalH; i++)
	{
		/*if (xfcountHost[i] == 1)
			ctr++;
		else if (xfcountHost[i] == 0)
		{

		}
		else
			cout << "incorrect value " << xfcountHost[i] << " at " << i << '\n';*/
		if (xfcountHost[i] > 0)
			ctr += xfcountHost[i];

		if (xfcountHost[i] > 1)
			MoreThanOne++;
		//if (xfcountHost[i] != 0)
		//	cout << "value in xfcountHost is: " << xfcountHost[i] << '\n';

		if (xfcountHost[i] > 0)
			occPixels++;
	}

	cout << MoreThanOne << " pixels have more than one frag (testing count)\n";

	//make sure 1s equal to vxs num
	cout << "sizes: " << vxPos->size() << ' ' << ctr << '\n'; //ctr should be equal to the vxpos size//
	cout << "occupied pixels: " << occPixels << " \n";


}

void DDS::CreateOffsetAndFragsVectors(bool DoSortFrags)
{
	float milliseconds;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	//set values
	SetOffsetVectorCuda(globalW * globalH, xfcount, xfoffset);

	FragsNum = GetFragsNumCuda(globalW * globalH, xfcount);
	if (debug_details)
		cout << "fragsnum: " << FragsNum << '\n';

	if (DoSortFrags) cudaMalloc((void**)&FragDepth, FragsNum * sizeof(float));
	cudaMalloc((void**)&FragVertex, FragsNum * sizeof(int));
	if (DoSortFrags) cudaMalloc((void**)&FragDepthPixel, FragsNum * sizeof(unsigned long long));

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	if (debug_details)
		cout << "***Create Offset and Frags vecs time: " << milliseconds << '\n';

}

void DDS::TestCreateOffset()
{
	int* xfcountHost = new int[globalW * globalH];
	cudaMemcpy(xfcountHost, xfcount, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);
	int* xfoffsetHost = new int[globalW * globalH];
	cudaMemcpy(xfoffsetHost, xfoffset, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);

	//int ctr = xfcountHost[0];
	int ctr = 0;
	for (int i = 1; i < globalW * globalH; i++)
	{
		ctr += xfcountHost[i - 1];
		if (xfoffsetHost[i] != ctr)
			cout << "problem at " << i << '\n';
	}

	//to make one frag, make loops from start to start, not from start to end//
	cout << "offset test: " << vxPos->size() << ' ' << ctr << '\n'; //equal when one frag per vertex//


}

void DDS::ProjectFrags(bool DoSortFrags)
{
	float milliseconds;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	//zero count again
	FillAllWithValue(xfcount, globalW * globalH, 0);

	//project using vbo data, count and offset
	ProjectFragsCuda(vxPos->size(), globalW, globalH, viewWidth, matrixVM, matrixPVM, vpos, xfcount, xfoffset, FragDepth, FragVertex, FragDepthPixel, DoSortFrags);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	if (debug_details)
		cout << "***ProjectFrags time: " << milliseconds << '\n';

}

void DDS::TestProjectFrags() //may be not the best test
{
	int newSum = GetFragsNumCuda(globalW * globalH, xfcount);
	cout << "fragsnum: " << newSum << ' ' << FragsNum << '\n';


	vector<bool>vtaken(vxPos->size(), false);
	int* vertexHost = new int[FragsNum];
	cudaMemcpy(vertexHost, FragVertex, vxPos->size() * sizeof(int), cudaMemcpyDeviceToHost);

	int ctr = 0;
	for (int i = 0; i < FragsNum; i++)
	{
		vtaken[vertexHost[i]] = true;
	}
	
	int pbm = 0;
	for (int i = 0; i < vxPos->size(); i++)
	{
		if (!vtaken[i])
			pbm++;
	}

	cout << "non projected vertices: " << pbm << '\n';


}

void DDS::SortFrags()
{

	float milliseconds;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	SortFragsCuda(FragsNum, FragDepth, FragVertex, FragDepthPixel);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	if (debug_details)
		cout << "***SortFrags time: " << milliseconds << '\n';

}

void DDS::TestSortFrags()
{

	/////////////
	//check (patch,pixel) are ordered. check depths are ordered//
	/////////////

	float* fragdepthHost = new float[FragsNum];
	cudaMemcpy(fragdepthHost, FragDepth, FragsNum * sizeof(float), cudaMemcpyDeviceToHost);
	unsigned long long* fragdepthpixelHost = new unsigned long long[FragsNum];
	cudaMemcpy(fragdepthpixelHost, FragDepthPixel, FragsNum * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 50; i++)
	{
		cout << fragdepthHost[i] << " " << fragdepthpixelHost[i] << '\n';
	}


	bool pxlproblem = false;
	bool dstproblem = false;
	float prevf = fragdepthHost[0];
	unsigned long long prevull = fragdepthpixelHost[0];
	for (int i = 1; i < FragsNum; i++)
	{
		float f = fragdepthHost[i];
		unsigned long long ull = fragdepthpixelHost[i];
		if (ull != prevull)
		{
			if (ull < prevull)
			{
				cout << "problem in the pixel order at " << i - 1 << " and " << i << '\n';
				pxlproblem = true;
			}
		}
		else
		{
			if (f < prevf)
			{
				cout << "problem in the depth order at " << i - 1 << " and " << i << '\n';
				cout << "values are " << prevull << ' ' << ull << ' ' << prevf << ' ' << f << '\n';
				dstproblem = true;
			}
		}

		prevull = ull;
		prevf = f;
	}
	if (!pxlproblem)
		cout << "no problems found in the sorted pixel list\n";
	if (!dstproblem)
		cout << "no problems found in the sorted depth list\n";

	
	/////////////
	//check all pixels with frags, exist in FragPatchPixel after sort//
	/////////////

	//get old count//
	int* xfcountHost = new int[globalW * globalH];
	cudaMemcpy(xfcountHost, xfcount, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);



	vector<bool> pixelFilled(globalW * globalH, false);
	for (int i = 0; i < globalW * globalH; i++)
	{
		if (xfcountHost[i] > 0)
			pixelFilled[i] = true;
	}

	vector<bool> pixelFilled2(globalW * globalH, false);
	for (int i = 0; i < FragsNum; i++)
	{
		unsigned long long patchpixel = fragdepthpixelHost[i];
		patchpixel = patchpixel >> 32;
		int pxl = patchpixel; //get it
		pixelFilled2[pxl] = true;
	}

	bool pbmFound = false;
	for (int i = 0; i < globalW * globalH; i++)
	{
		if ((pixelFilled[i] && !pixelFilled2[i]) || (!pixelFilled[i] && pixelFilled2[i]))
			cout << "problem at pixel " << i << '\n';
		if ((pixelFilled[i] && !pixelFilled2[i]) || (!pixelFilled[i] && pixelFilled2[i]))
			pbmFound = true;
	}
	if (!pbmFound)
		cout << "pixels with frags are same frags saved in depthpixel\n";

	int ccc1 = 0, ccc2 = 0;
	for (int i = 0; i < globalW * globalH; i++)
	{
		if (pixelFilled[i])
			ccc1++;
		if (pixelFilled2[i])
			ccc2++;
	}
	cout << "pixels filled: " << ccc1 << ' ' << ccc2 << '\n';

	/////////////
	//check all pixels with in fragdepthpixel, are equivalent to pixel of subarray//
	/////////////

	int* xfoffsetHost = new int[globalW * globalH];
	cudaMemcpy(xfoffsetHost, xfoffset, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);

	pbmFound = false;
	for (int i = 0; i < globalW * globalH; i++)
	{
		bool pxlPbm = false;
		for (int j = xfoffsetHost[i]; j < xfoffsetHost[i] + xfcountHost[i]; j++)
		{
			unsigned long long patchpixel = fragdepthpixelHost[j];
			patchpixel = patchpixel >> 32;
			int pxl = patchpixel; //get it
			if (pxl != i)
				pxlPbm = true;
				
		}
		if (pxlPbm)
		{
			pbmFound = true;
			cout << "problem at pixel " << i << ", pixels: ";
			for (int j = xfoffsetHost[i]; j < xfoffsetHost[i] + xfcountHost[i]; j++)
			{
				unsigned long long patchpixel = fragdepthpixelHost[j];
				patchpixel = patchpixel >> 32;
				int pxl = patchpixel; //get it
				cout << pxl << ' ';
			}
			cout << '\n';
		}

	}
	if (!pbmFound)
		cout << "All pixels saved in fragdepthpixelHost are equivalent to pixel of the subarray\n";


}



void DDS::FreeMemory()
{
	float milliseconds;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	cudaFree(vpos);

	cudaFree(matrixVM);
	cudaFree(matrixPVM);

	cudaFree(xfcount);
	cudaFree(xfoffset);
	
	cudaFree(FragDepth);
	cudaFree(FragVertex);
	cudaFree(FragDepthPixel);
	

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	if (debug_details)
		cout << "***FreeMemory time: " << milliseconds << '\n';

}

float DDS::BuildDDS(bool DoSortFrags, bool dd)
{
	debug_details = dd;

	float milliseconds;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	PrepareInput();
	CountFrags();
	//TestCountFrags();
	CreateOffsetAndFragsVectors(DoSortFrags);
	//TestCreateOffset();
	ProjectFrags(DoSortFrags);
	//TestProjectFrags();
	if(DoSortFrags)
		SortFrags();
	
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "***total DDS time: " << milliseconds << '\n';

	cout << "---------\n";

	//CopyPilesToCPU();
	//TestCopyPilesToCPU();
	//FreeMemory();

	return milliseconds;
}

void DDS::findKNN(int k, float SearchRad, bool DoSortFrags)
{
	if (DoSortFrags)
		findKNNwithSort(k, SearchRad);
	else
		findKNN(k, SearchRad);
}

DDS::~DDS()
{
	FreeMemory();
}

