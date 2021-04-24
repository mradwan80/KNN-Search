#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/device_ptr.h>
#include<thrust/sort.h>
#include<thrust/sequence.h>
#include<thrust/gather.h>
#include<thrust/count.h>
#include <thrust/execution_policy.h>
#include<thrust/copy.h>
#include "DDS.h"


__global__ void CountFragsKernel(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, int* xfcount)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;

	if (v < vxNum)
	{

		//get vertex//
		float x = vpos[3 * v + 0];
		float y = vpos[3 * v + 1];
		float z = vpos[3 * v + 2];
		float w = 1.0;

		float posXpvm = pvmMat[0] * x + pvmMat[4] * y + pvmMat[8] * z + pvmMat[12] * w;
		float posYpvm = pvmMat[1] * x + pvmMat[5] * y + pvmMat[9] * z + pvmMat[13] * w;
		float posWpvm = pvmMat[3] * x + pvmMat[7] * y + pvmMat[11] * z + pvmMat[15] * w;

		int xscreen = (int)(((posXpvm / posWpvm) / 2 + 0.5) * globalW);
		int yscreen = (int)(((posYpvm / posWpvm) / 2 + 0.5) * globalH);

		int pxl = xscreen + yscreen * globalW;

		if (pxl >= 0 && pxl < globalW * globalH)
			atomicAdd(&xfcount[pxl], 1);

	}
}


void CountFragsCuda(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, int* xfcount)
{
	CountFragsKernel << <vxNum / 256 + 1, 256 >> > (vxNum, globalW, globalH, viewWidth, vmMat, pvmMat, vpos, xfcount);
}


void SetOffsetVectorCuda(int pxNum, int* xfcount, int* xfoffset)
{
	thrust::device_ptr<int> o = thrust::device_pointer_cast(xfoffset);
	thrust::device_ptr<int> c = thrust::device_pointer_cast(xfcount);

	//call thrust function
	thrust::exclusive_scan(c, c + pxNum, o);
}

int GetFragsNumCuda(int vxNum, int* xfcount)
{
	thrust::device_ptr<int> c = thrust::device_pointer_cast(xfcount);

	//get count of xfcount//
	int FragsNum = thrust::reduce(c, c + vxNum, (int)0, thrust::plus<int>());

	return FragsNum;
}

__device__
unsigned long long GeneratePixelDepthKey(int pixel, float depth)
{
	unsigned long long result = pixel;
	result = result << 32;

	//unsigned long long result=0;

	const float lineParameter = depth;
	//uint converted_key = *((uint *)&lineParameter);
	unsigned int converted_key = *((unsigned int*)&lineParameter);
	const unsigned int mask = ((converted_key & 0x80000000) ? 0xffffffff : 0x80000000);
	converted_key ^= mask;

	result |= (unsigned long long)(converted_key);

	return result;

}

__global__ void ProjectFragsForSortKernel(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, int* xfcount, int* xfoffset, float* FragDepth, int* FragVertex, unsigned long long* FragDepthPixel)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;

	if (v < vxNum)
	{

		//get vertex//
		float x = vpos[3 * v + 0];
		float y = vpos[3 * v + 1];
		float z = vpos[3 * v + 2];
		float w = 1.0;

		float posXpvm = pvmMat[0] * x + pvmMat[4] * y + pvmMat[8] * z + pvmMat[12] * w;
		float posYpvm = pvmMat[1] * x + pvmMat[5] * y + pvmMat[9] * z + pvmMat[13] * w;
		float posZpvm = pvmMat[2] * x + pvmMat[6] * y + pvmMat[10] * z + pvmMat[14] * w;
		float posWpvm = pvmMat[3] * x + pvmMat[7] * y + pvmMat[11] * z + pvmMat[15] * w;

		int xscreen = (int)(((posXpvm / posWpvm) / 2 + 0.5) * globalW);
		int yscreen = (int)(((posYpvm / posWpvm) / 2 + 0.5) * globalH);

		float posZvm = vmMat[2] * x + vmMat[6] * y + vmMat[10] * z + vmMat[14] * w;

		float depth = posZvm;

		int pxl = xscreen + yscreen * globalW; //from x and y

		int index, offset;
		offset = xfoffset[pxl];
		index = atomicAdd(&xfcount[pxl], 1) + offset;

		FragDepth[index] = depth;
		FragVertex[index] = v;
		FragDepthPixel[index] = GeneratePixelDepthKey(pxl, depth);



	}

}


__global__ void ProjectFragsKernel(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, int* xfcount, int* xfoffset, int* FragVertex)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;

	if (v < vxNum)
	{

		//get vertex//
		float x = vpos[3 * v + 0];
		float y = vpos[3 * v + 1];
		float z = vpos[3 * v + 2];
		float w = 1.0;

		float posXpvm = pvmMat[0] * x + pvmMat[4] * y + pvmMat[8] * z + pvmMat[12] * w;
		float posYpvm = pvmMat[1] * x + pvmMat[5] * y + pvmMat[9] * z + pvmMat[13] * w;
		float posZpvm = pvmMat[2] * x + pvmMat[6] * y + pvmMat[10] * z + pvmMat[14] * w;
		float posWpvm = pvmMat[3] * x + pvmMat[7] * y + pvmMat[11] * z + pvmMat[15] * w;

		int xscreen = (int)(((posXpvm / posWpvm) / 2 + 0.5) * globalW);
		int yscreen = (int)(((posYpvm / posWpvm) / 2 + 0.5) * globalH);

		float posZvm = vmMat[2] * x + vmMat[6] * y + vmMat[10] * z + vmMat[14] * w;

		float depth = posZvm;

		int pxl = xscreen + yscreen * globalW; //from x and y

		int index, offset;
		offset = xfoffset[pxl];
		index = atomicAdd(&xfcount[pxl], 1) + offset;

		FragVertex[index] = v;


	}

}

void ProjectFragsCuda(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, int* xfcount, int* xfoffset, float* FragDepth, int* FragVertex, unsigned long long* FragDepthPixel, bool DoSortFrags)
{
	if(DoSortFrags)
		ProjectFragsForSortKernel << <vxNum / 256 + 1, 256 >> > (vxNum, globalW, globalH, viewWidth, vmMat, pvmMat, vpos, xfcount, xfoffset, FragDepth, FragVertex, FragDepthPixel);
	else
		ProjectFragsKernel << <vxNum / 256 + 1, 256 >> > (vxNum, globalW, globalH, viewWidth, vmMat, pvmMat, vpos, xfcount, xfoffset, FragVertex);
}

//works fine as long as #frags is ok. when not, need reformulating, so that not all buffers are allocated at same time//
void SortFragsCuda(int FragsNum, float* FragDepth, int* FragVertex, unsigned long long* FragDepthPixel)
{
	//device pointers//
	thrust::device_ptr<float> fd = thrust::device_pointer_cast(FragDepth);
	thrust::device_ptr<int> fv = thrust::device_pointer_cast(FragVertex);
	thrust::device_ptr<unsigned long long> fdp = thrust::device_pointer_cast(FragDepthPixel);

	//tmp buffers for thrust::gather//
	float* FragDepthTmp;
	int* FragVertexTmp;
	cudaMalloc((void**)&FragDepthTmp, FragsNum * sizeof(float));
	cudaMalloc((void**)&FragVertexTmp, FragsNum * sizeof(int));
	thrust::device_ptr<float> fdt = thrust::device_pointer_cast(FragDepthTmp);
	thrust::device_ptr<int> fvt = thrust::device_pointer_cast(FragVertexTmp);
	
	//init an index buffer//
	unsigned int* FragIndex;
	cudaMalloc((void**)&FragIndex, FragsNum * sizeof(unsigned int));
	thrust::device_ptr<unsigned int> fi = thrust::device_pointer_cast(FragIndex);
	thrust::sequence(fi, fi + FragsNum, 0);


	//sort depth and index//
	thrust::sort_by_key(fdp, fdp + FragsNum, fi);


	//change all other arrays based on the sorted index//
	thrust::gather(fi, fi + FragsNum, fd, fdt);
	thrust::gather(fi, fi + FragsNum, fv, fvt);
	cudaMemcpy(FragDepth, FragDepthTmp, FragsNum * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(FragVertex, FragVertexTmp, FragsNum * sizeof(int), cudaMemcpyDeviceToDevice);

}



//use a template?
void FillAllWithValue(int* arr, int sz, int val)
{

	thrust::device_ptr<int> d = thrust::device_pointer_cast(arr);
	thrust::fill(d, d + sz, val);

}

void FillAllWithValue(float* arr, int sz, float val)
{

	thrust::device_ptr<float> d = thrust::device_pointer_cast(arr);
	thrust::fill(d, d + sz, val);

}

void FillAllWithValue(bool* arr, int sz, bool val)
{

	thrust::device_ptr<bool> d = thrust::device_pointer_cast(arr);
	thrust::fill(d, d + sz, val);

}

void FillAllWithValue(unsigned long long* arr, int sz, unsigned long long val)
{
	thrust::device_ptr<unsigned long long> d = thrust::device_pointer_cast(arr);
	thrust::fill(d, d + sz, val);
}

