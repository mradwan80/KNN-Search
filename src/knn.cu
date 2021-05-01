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

__global__ void CopyCountsKernel(int qnum, int len, int globalW, int globalH, float* pvmMat, float* vpos, int* xfcount, bool* pixelIn, int* sncount)
{
	int qspxl = blockIdx.x * blockDim.x + threadIdx.x;

	if (qspxl < qnum * len * len)
	{
		int q = qspxl / (len * len);
		int spxl = qspxl % (len * len);
		int sx = spxl % len;
		int sy = spxl / len;

		if (!pixelIn[spxl])
			return;

		//get vertex//
		float x = vpos[3 * q + 0];
		float y = vpos[3 * q + 1];
		float z = vpos[3 * q + 2];
		float w = 1.0;

		float posXpvm = pvmMat[0] * x + pvmMat[4] * y + pvmMat[8] * z + pvmMat[12] * w;
		float posYpvm = pvmMat[1] * x + pvmMat[5] * y + pvmMat[9] * z + pvmMat[13] * w;
		float posZpvm = pvmMat[2] * x + pvmMat[6] * y + pvmMat[10] * z + pvmMat[14] * w;
		float posWpvm = pvmMat[3] * x + pvmMat[7] * y + pvmMat[11] * z + pvmMat[15] * w;

		//exact pixel of q//
		int qxscreen = (int)(((posXpvm / posWpvm) / 2 + 0.5) * globalW);
		int qyscreen = (int)(((posYpvm / posWpvm) / 2 + 0.5) * globalH);

		//pixel in the square, based on thread//
		int xscreen = (qxscreen - len/2) + sx;
		int yscreen = (qyscreen - len/2) + sy;
		int pxl = xscreen + yscreen * globalW;

		if (xscreen<0 || xscreen>globalW - 1 || yscreen<0 || yscreen>globalH - 1)
			return;

		//copy counts//
		sncount[qspxl] = xfcount[pxl];

	}

}

void CopyCountsCuda(int qnum, int len, int globalW, int globalH, float* pvmMat, float* vpos, int* xfcount, bool* pixelIn, int* sncount)
{
	CopyCountsKernel << < (qnum * len * len) / 256 + 1, 256 >> > (qnum, len, globalW, globalH, pvmMat, vpos, xfcount, pixelIn, sncount);

}

void CreateNbsOffsetArrayCuda(int n, int* sncount, int* snoffset)
{
	thrust::device_ptr<int> o = thrust::device_pointer_cast(snoffset);
	thrust::device_ptr<int> c = thrust::device_pointer_cast(sncount);

	//call thrust function
	thrust::exclusive_scan(c, c + n, o);
}


int SumNbsCuda(int n, int* sncount)
{

	thrust::device_ptr<int> c = thrust::device_pointer_cast(sncount);

	//get count of xfcount//
	int NbsNum = thrust::reduce(c, c + n, (int)0, thrust::plus<int>());

	return NbsNum;
}

__device__
unsigned long long GenerateVertexDistKey(int vertex, float dist)
{
	unsigned long long result = vertex;
	result = result << 32;

	//unsigned long long result=0;

	const float lineParameter = dist;
	//uint converted_key = *((uint *)&lineParameter);
	unsigned int converted_key = *((unsigned int*)&lineParameter);
	const unsigned int mask = ((converted_key & 0x80000000) ? 0xffffffff : 0x80000000);
	converted_key ^= mask;

	result |= (unsigned long long)(converted_key);

	return result;

}


__global__
void FillDistanceKernel(int qnum, int len, int globalW, int globalH, float* pvmMat, float* vpos, int* xfcount, int* xfoffset, int* FragVertex, bool* pixelIn, int* snoffset, int* NbVertex, unsigned long long* NbVertexDist)
{
	int qspxl = blockIdx.x * blockDim.x + threadIdx.x;

	if (qspxl < qnum * len * len)
	{
		int q = qspxl / (len * len);
		int spxl = qspxl % (len * len);
		int sx = spxl % len;
		int sy = spxl / len;

		if (!pixelIn[spxl])
			return;

		//get vertex//
		float x = vpos[3 * q + 0];
		float y = vpos[3 * q + 1];
		float z = vpos[3 * q + 2];
		float w = 1.0;

		float posXpvm = pvmMat[0] * x + pvmMat[4] * y + pvmMat[8] * z + pvmMat[12] * w;
		float posYpvm = pvmMat[1] * x + pvmMat[5] * y + pvmMat[9] * z + pvmMat[13] * w;
		float posZpvm = pvmMat[2] * x + pvmMat[6] * y + pvmMat[10] * z + pvmMat[14] * w;
		float posWpvm = pvmMat[3] * x + pvmMat[7] * y + pvmMat[11] * z + pvmMat[15] * w;

		//exact pixel of q//
		int qxscreen = (int)(((posXpvm / posWpvm) / 2 + 0.5) * globalW);
		int qyscreen = (int)(((posYpvm / posWpvm) / 2 + 0.5) * globalH);

		//pixel in the square, based on thread//
		int xscreen = (qxscreen - len/2) + sx;
		int yscreen = (qyscreen - len/2) + sy;
		int pxl = xscreen + yscreen * globalW;

		if (xscreen<0 || xscreen>globalW - 1 || yscreen<0 || yscreen>globalH - 1)
			return;

		int offset = xfoffset[pxl];
		for (int v2 = 0; v2 < xfcount[pxl]; v2++)
		{
			int vx2 = FragVertex[v2 + offset];
			float x2 = vpos[3 * vx2 + 0];
			float y2 = vpos[3 * vx2 + 1];
			float z2 = vpos[3 * vx2 + 2];

			float dist = (x - x2) * (x - x2) + (y - y2) * (y - y2) + (z - z2) * (z - z2);

			NbVertex[snoffset[qspxl] + v2] = vx2;
			NbVertexDist[snoffset[qspxl] + v2] = GenerateVertexDistKey(q, dist);


		}

	}

}

void FillDistanceCuda(int qnum, int len, int globalW, int globalH, float* pvmMat, float* vpos, int* xfcount, int* xfoffset, int* FragVertex, bool* pixelIn, int* snoffset, int* NbVertex, unsigned long long* NbVertexDist)
{

	FillDistanceKernel << < (qnum * len * len) / 256 + 1, 256 >> > (qnum, len, globalW, globalH, pvmMat, vpos, xfcount, xfoffset, FragVertex, pixelIn, snoffset, NbVertex, NbVertexDist);


}

void SortNeighborsCuda(int NbsNum, int* NbVertex, unsigned long long* NbVertexDist)
{

	//device pointers//
	thrust::device_ptr<int> fv = thrust::device_pointer_cast(NbVertex);
	thrust::device_ptr<unsigned long long> fvd = thrust::device_pointer_cast(NbVertexDist);

	//tmp buffers for thrust::gather//
	int* NbVertexTmp;
	cudaMalloc((void**)&NbVertexTmp, NbsNum * sizeof(int));
	thrust::device_ptr<int> fvt = thrust::device_pointer_cast(NbVertexTmp);

	//init an index buffer//
	unsigned int* NbIndex;
	cudaMalloc((void**)&NbIndex, NbsNum * sizeof(unsigned int));
	thrust::device_ptr<unsigned int> fi = thrust::device_pointer_cast(NbIndex);
	thrust::sequence(fi, fi + NbsNum, 0);


	//sort depth and index//
	thrust::sort_by_key(fvd, fvd + NbsNum, fi);


	//change all other arrays based on the sorted index//
	thrust::gather(fi, fi + NbsNum, fv, fvt);
	cudaMemcpy(NbVertex, NbVertexTmp, NbsNum * sizeof(int), cudaMemcpyDeviceToDevice);

}

void CopyKNeighborsCuda(int k, float SearchRad, int qnum, int len, int* sncount, int NbsNum, int* NbVertex, int vnum, float* vpos, vector<vector<int>>& Nbs)
{
	int* NbVertexHost = new int[NbsNum];
	cudaMemcpy(NbVertexHost, NbVertex, NbsNum * sizeof(int), cudaMemcpyDeviceToHost);
	int* sncountHost = new int[qnum * len * len];
	cudaMemcpy(sncountHost, sncount, qnum * len * len * sizeof(int), cudaMemcpyDeviceToHost);
	float* vposHost = new float[vnum * 3];
	cudaMemcpy(vposHost, vpos, vnum * 3 * sizeof(float), cudaMemcpyDeviceToHost);

	int offset = 0;
	for (int q = 0; q < qnum; q++)
	{
		int start = len * len * q;
		int end = start + len * len;
		int acc = 0;
		for (int i = start; i < end; i++)
		{
			acc += sncountHost[i];
		}
		
		int copyn;
		if (acc < k)
			copyn = acc;
		else
			copyn = k;

		int counter = 0;
		for (int i = 0; i < copyn; i++)
		{
			int vx = NbVertexHost[offset + i];


			float dist = sqrt((vposHost[3 * q + 0] - vposHost[3 * vx + 0]) * (vposHost[3 * q + 0] - vposHost[3 * vx + 0]) + (vposHost[3 * q + 1] - vposHost[3 * vx + 1]) * (vposHost[3 * q + 1] - vposHost[3 * vx + 1]) + (vposHost[3 * q + 2] - vposHost[3 * vx + 2]) * (vposHost[3 * q + 2] - vposHost[3 * vx + 2]));

			if (dist < SearchRad)
				Nbs[q][counter++] = vx;
		}
		offset += acc;

	}

}


