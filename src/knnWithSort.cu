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

__global__ void CopyCountsKernelS(int qnum, int len, float searchRad, float cellWidth, int globalW, int globalH, float* vmMat, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragDepth, int* qncount)
{
	int q = blockIdx.x * blockDim.x + threadIdx.x;

	while (q < qnum)
	{

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

		int pcount = 0;
		for (int sx = 0; sx < len; sx++)
		{
			for (int sy = 0; sy < len; sy++)
			{

				//pixel in the square, based on thread//
				int xscreen = (qxscreen - len / 2) + sx;
				int yscreen = (qyscreen - len / 2) + sy;
				int pxl = xscreen + yscreen * globalW;

				if (xscreen<0 || xscreen>globalW - 1 || yscreen<0 || yscreen>globalH - 1)
					continue;

				if (xfcount[pxl] == 0)
					continue;

				//decide what is the distance range allowed in the cetain pixel (wrt the qpixel)
				float qdpt = vmMat[2] * qx + vmMat[6] * qy + vmMat[10] * qz + vmMat[14] * qw;
				float xdiff = (abs(xscreen - qxscreen) + 0) * cellWidth;
				float ydiff = (abs(yscreen - qyscreen) + 0) * cellWidth;
				float pxldiffsqr = xdiff * xdiff + ydiff * ydiff;
				//float pxlradsqr = searchRad * searchRad - pxldiffsqr;
				float pxlradsqr = searchRad * searchRad * 1.44 - pxldiffsqr;

				//go through distances of vxs in this pixel
				int offset = xfoffset[pxl];
				int count = xfcount[pxl];
				for (int f = 0; f < count; f++)
				{
					int findex = offset + f;
					float pdpt = FragDepth[findex];
					float zdiff = abs(qdpt - pdpt);

					if (zdiff * zdiff <= pxlradsqr)
						pcount++;

				}
			}
		}

		qncount[q] = pcount;

		q += gridDim.x * blockDim.x;
		
	}

}



void CopyCountsCudaS(int qnum, int len, float searchRad, float cellWidth, int globalW, int globalH, float* vmMat, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragDepth, int* qncount)
{
	int gridsize = qnum / blocksize + 1;
	if (gridsize > maxblocks) gridsize = maxblocks;

	CopyCountsKernelS << < gridsize, blocksize >> > (qnum, len, searchRad, cellWidth, globalW, globalH, vmMat, pvmMat, xfcount, xfoffset, vpos, FragDepth, qncount);

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
void FillDistanceKernelS(int qnum, int len, float searchRad, float cellWidth, int globalW, int globalH, float* vmMat, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragX, float* FragY, float* FragZ, int* FragVertex, float* FragDepth, int* qnoffset, int* NbVertex, unsigned long long* NbVertexDist)
{
	int q = blockIdx.x * blockDim.x + threadIdx.x;

	if (q < qnum)
	{

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

		int qoffset = qnoffset[q];

		int pcount = 0;
		for (int sx = 0; sx < len; sx++)
		{
			for (int sy = 0; sy < len; sy++)
			{
				//pixel in the square, based on thread//
				int xscreen = (qxscreen - len / 2) + sx;
				int yscreen = (qyscreen - len / 2) + sy;
				int pxl = xscreen + yscreen * globalW;

				if (xscreen<0 || xscreen>globalW - 1 || yscreen<0 || yscreen>globalH - 1)
					continue;

				if (xfcount[pxl] == 0)
					continue;

				//decide what is the distance range allowed in the cetain pixel (wrt the qpixel)
				float qdpt = vmMat[2] * qx + vmMat[6] * qy + vmMat[10] * qz + vmMat[14] * qw;
				float xdiff = (abs(xscreen - qxscreen) + 0) * cellWidth;
				float ydiff = (abs(yscreen - qyscreen) + 0) * cellWidth;
				float pxldiffsqr = xdiff * xdiff + ydiff * ydiff;
				//float pxlradsqr = searchRad * searchRad - pxldiffsqr;
				float pxlradsqr = searchRad * searchRad * 1.44 - pxldiffsqr;

				//go through distances of vxs in this pixel
				int offset = xfoffset[pxl];
				int count = xfcount[pxl];
				for (int f = 0; f < count; f++)
				{
					int findex = offset + f;

					float pdpt = FragDepth[findex];
					float zdiff = abs(qdpt - pdpt);

					if (zdiff * zdiff <= pxlradsqr)
					{
						int vx = FragVertex[findex];
						float x = FragX[findex];
						float y = FragY[findex];
						float z = FragZ[findex];
						float dist = (qx - x) * (qx - x) + (qy - y) * (qy - y) + (qz - z) * (qz - z);

						NbVertex[qoffset + pcount] = vx;
						NbVertexDist[qoffset + pcount] = GenerateVertexDistKeyS(q, dist);

						pcount++;
					}

				}
			}
		}

		q += gridDim.x * blockDim.x;

	}
	
}

void FillDistanceCudaS(int qnum, int len, float searchRad, float cellWidth, int globalW, int globalH, float* vmMat, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragX, float* FragY, float* FragZ, int* FragVertex, float* FragDepth, int* qnoffset, int* NbVertex, unsigned long long* NbVertexDist)
{
	int gridsize = qnum / blocksize + 1;
	if (gridsize > maxblocks) gridsize = maxblocks;

	FillDistanceKernelS << < gridsize, blocksize >> > (qnum, len, searchRad, cellWidth, globalW, globalH, vmMat, pvmMat, xfcount, xfoffset, vpos, FragX, FragY, FragZ, FragVertex, FragDepth, qnoffset, NbVertex, NbVertexDist);


}