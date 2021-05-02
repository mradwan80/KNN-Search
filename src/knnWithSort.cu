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

__global__ void CopyCountsKernelS(int qnum, int len, float searchRad, float cellWidth, int globalW, int globalH, float* vmMat, float* pvmMat, float* vpos, int* xfcount, int* xfoffset, float* FragDepth, bool* pixelIn, int* sncount)
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
		float qx = vpos[3 * q + 0];
		float qy = vpos[3 * q + 1];
		float qz = vpos[3 * q + 2];
		float qw = 1.0;

		float posXpvm = pvmMat[0] * qx + pvmMat[4] * qy + pvmMat[8] * qz + pvmMat[12] * qw;
		float posYpvm = pvmMat[1] * qx + pvmMat[5] * qy + pvmMat[9] * qz + pvmMat[13] * qw;
		float posZpvm = pvmMat[2] * qx + pvmMat[6] * qy + pvmMat[10] * qz + pvmMat[14] * qw;
		float posWpvm = pvmMat[3] * qx + pvmMat[7] * qy + pvmMat[11] * qz + pvmMat[15] * qw;

		//exact pixel of q//
		int qxscreen = (int)(((posXpvm / posWpvm) / 2 + 0.5) * globalW);
		int qyscreen = (int)(((posYpvm / posWpvm) / 2 + 0.5) * globalH);

		//pixel in the square, based on thread//
		int xscreen = (qxscreen - len / 2) + sx;
		int yscreen = (qyscreen - len / 2) + sy;
		int pxl = xscreen + yscreen * globalW;

		if (xscreen<0 || xscreen>globalW - 1 || yscreen<0 || yscreen>globalH - 1)
			return;

		if (xfcount[pxl] == 0)
			return;

		//decide what is the distance range allowed in the cetain pixel (wrt the qpixel)
		float qdpt = vmMat[2] * qx + vmMat[6] * qy + vmMat[10] * qz + vmMat[14] * qw;
		float xdiff = (abs(xscreen - qxscreen) + 0) * cellWidth;
		float ydiff = (abs(yscreen - qyscreen) + 0) * cellWidth;
		float pxldiffsqr = xdiff * xdiff + ydiff * ydiff;
		//float pxlradsqr = searchRad * searchRad - pxldiffsqr;
		float pxlradsqr = searchRad * searchRad * 1.44 - pxldiffsqr;

		//go through distances of vxs in this pixel
		int pcount = 0;
		int offset = xfoffset[pxl];
		for (int j = 0; j < xfcount[pxl]; j++)
		{
			int pindex = offset + j;
			float pdpt = FragDepth[pindex];
			float zdiff = abs(qdpt - pdpt);

			if (zdiff * zdiff <= pxlradsqr)
				pcount++;

		}

		sncount[qspxl] = pcount;
		
	}

}



void CopyCountsCudaS(int qnum, int len, float searchRad, float cellWidth, int globalW, int globalH, float* vmMat, float* pvmMat, float* vpos, int* xfcount, int* xfoffset, float* FragDepth, bool* pixelIn, int* sncount)
{
	CopyCountsKernelS << < (qnum * len * len) / 256 + 1, 256 >> > (qnum, len, searchRad, cellWidth, globalW, globalH, vmMat, pvmMat, vpos, xfcount, xfoffset, FragDepth, pixelIn, sncount);

}



__device__
unsigned long long GenerateVertexDistKeyS(int vertex, float dist)
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
void FillDistanceKernelS(int qnum, int len, float searchRad, float cellWidth, int globalW, int globalH, float* vmMat, float* pvmMat, float* vpos, int* xfcount, int* xfoffset, int* FragVertex, float* FragDepth, bool* pixelIn, int* sncount, int* snoffset, int* NbVertex, unsigned long long* NbVertexDist)
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
		float qx = vpos[3 * q + 0];
		float qy = vpos[3 * q + 1];
		float qz = vpos[3 * q + 2];
		float qw = 1.0;

		float posXpvm = pvmMat[0] * qx + pvmMat[4] * qy + pvmMat[8] * qz + pvmMat[12] * qw;
		float posYpvm = pvmMat[1] * qx + pvmMat[5] * qy + pvmMat[9] * qz + pvmMat[13] * qw;
		float posZpvm = pvmMat[2] * qx + pvmMat[6] * qy + pvmMat[10] * qz + pvmMat[14] * qw;
		float posWpvm = pvmMat[3] * qx + pvmMat[7] * qy + pvmMat[11] * qz + pvmMat[15] * qw;

		//exact pixel of q//
		int qxscreen = (int)(((posXpvm / posWpvm) / 2 + 0.5) * globalW);
		int qyscreen = (int)(((posYpvm / posWpvm) / 2 + 0.5) * globalH);

		//pixel in the square, based on thread//
		int xscreen = (qxscreen - len / 2) + sx;
		int yscreen = (qyscreen - len / 2) + sy;
		int pxl = xscreen + yscreen * globalW;

		if (xscreen<0 || xscreen>globalW - 1 || yscreen<0 || yscreen>globalH - 1)
			return;

		if (xfcount[pxl] == 0)
			return;

		//decide what is the distance range allowed in the cetain pixel (wrt the qpixel)
		float qdpt = vmMat[2] * qx + vmMat[6] * qy + vmMat[10] * qz + vmMat[14] * qw;
		float xdiff = (abs(xscreen - qxscreen) + 0) * cellWidth;
		float ydiff = (abs(yscreen - qyscreen) + 0) * cellWidth;
		float pxldiffsqr = xdiff * xdiff + ydiff * ydiff;
		//float pxlradsqr = searchRad * searchRad - pxldiffsqr;
		float pxlradsqr = searchRad * searchRad * 1.44 - pxldiffsqr;

		//go through distances of vxs in this pixel
		int pcount = 0;
		int offset = xfoffset[pxl];
		for (int j = 0; j < xfcount[pxl]; j++)
		{
			int pindex = offset + j;
			
			float pdpt = FragDepth[pindex];
			float zdiff = abs(qdpt - pdpt);

			if (zdiff * zdiff <= pxlradsqr)
			{
				int vx = FragVertex[pindex];
				float x = vpos[3 * vx + 0];
				float y = vpos[3 * vx + 1];
				float z = vpos[3 * vx + 2];
				float dist = (qx - x) * (qx - x) + (qy - y) * (qy - y) + (qz - z) * (qz - z);

				int pos = atomicAdd(&sncount[qspxl], 1);

				NbVertex[snoffset[qspxl] + pos] = vx;
				NbVertexDist[snoffset[qspxl] + pos] = GenerateVertexDistKeyS(q, dist);
			}

		}

	}

}

void FillDistanceCudaS(int qnum, int len, float searchRad, float cellWidth, int globalW, int globalH, float* vmMat, float* pvmMat, float* vpos, int* xfcount, int* xfoffset, int* FragVertex, float* FragDepth, bool* pixelIn, int* sncount, int* snoffset, int* NbVertex, unsigned long long* NbVertexDist)
{

	FillDistanceKernelS << < (qnum * len * len) / 256 + 1, 256 >> > (qnum, len, searchRad, cellWidth, globalW, globalH, vmMat, pvmMat, vpos, xfcount, xfoffset, FragVertex, FragDepth, pixelIn, sncount, snoffset, NbVertex, NbVertexDist);


}