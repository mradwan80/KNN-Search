#include "DDS.h"
#include <chrono>
#include <iostream>
#include <glm/gtc/type_ptr.hpp>
#include<thrust/device_ptr.h>
#include<thrust/sort.h>
#include <thrust/execution_policy.h>
using namespace std;

void checkCUDAError(const char* msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		getchar();
		exit(EXIT_FAILURE);
	}
}

DDS::DDS() {}

DDS::DDS(int w, int h, float viewWidthI, vector<pointCoords>* Pos, glm::mat4 vmMatI, glm::mat4 pvmMatI)
{
	globalW = w;
	globalH = h;
	viewWidth = viewWidthI;

	vxPos = Pos;

	vmMat = vmMatI;
	pvmMat = pvmMatI;

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


void DDS::GetFragsCoords()
{
	cudaMalloc((void**)&FragX, FragsNum * sizeof(float));
	cudaMalloc((void**)&FragY, FragsNum * sizeof(float));
	cudaMalloc((void**)&FragZ, FragsNum * sizeof(float));

	GetFragsCoordsCuda(FragsNum, FragVertex, vpos, FragX, FragY, FragZ);

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
	CountFrags();		checkCUDAError("Count failed");
	CreateOffsetAndFragsVectors(DoSortFrags);		checkCUDAError("Create Vectors failed");
	ProjectFrags(DoSortFrags);		checkCUDAError("Project failed");
	if (DoSortFrags)
	{
		SortFrags();	checkCUDAError("Sort failed");
	}
	
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "***total DDS time: " << milliseconds << '\n';

	cout << "---------\n";

	//CopyPilesToCPU();
	//FreeMemory();

	return milliseconds;
}

void DDS::findKNN(int k, float SearchRad, bool DoSortFrags)
{
	if (DoSortFrags)
		kNNsearchWithSort(k, SearchRad);
	else
		kNNsearch(k, SearchRad);
}

DDS::~DDS()
{
	FreeMemory();
}

