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


__global__ void CopyCountsKernel(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragX, float* FragY, float* FragZ, int* qncount)
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

				int offset = xfoffset[pxl];
				int count = xfcount[pxl];
				for (int f = 0; f < count; f++)
				{
					int findex = offset + f;

					//get pos
					float x = FragX[findex];
					float y = FragY[findex];
					float z = FragZ[findex];

					float dist = (x - qx) * (x - qx) + (y - qy) * (y - qy) + (z - qz) * (z - qz); //calc distance//

					if (dist <= searchRad * searchRad)
						pcount++;

				}


			}
		}
		
		
		qncount[q] = pcount;

		q += gridDim.x * blockDim.x;
	}

}

void CopyCountsCuda(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragX, float* FragY, float* FragZ, int* qncount)
{
	int gridsize = qnum / blocksize + 1;
	if (gridsize > maxblocks) gridsize = maxblocks;

	CopyCountsKernel << < gridsize, blocksize >> > (qnum, len, searchRad, globalW, globalH, pvmMat, xfcount, xfoffset, vpos, FragX, FragY, FragZ, qncount);
}

void CreateNbsOffsetArrayCuda(int n, int* qncount, int* qnoffset)
{
	thrust::device_ptr<int> o = thrust::device_pointer_cast(qnoffset);
	thrust::device_ptr<int> c = thrust::device_pointer_cast(qncount);

	//call thrust function
	thrust::exclusive_scan(c, c + n, o);
}


int SumNbsCuda(int n, int* qncount)
{

	thrust::device_ptr<int> c = thrust::device_pointer_cast(qncount);

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


__global__ void FillDistanceKernel(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragX, float* FragY, float* FragZ, int* FragVertex, int* qnoffset, int* NbVertex, unsigned long long* NbVertexDist)
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

				int offset = xfoffset[pxl];
				int count = xfcount[pxl];
				for (int f = 0; f < count; f++)
				{
					int findex = offset + f;
					
					//get pos
					float x = FragX[findex];
					float y = FragY[findex];
					float z = FragZ[findex];

					float dist = (x - qx) * (x - qx) + (y - qy) * (y - qy) + (z - qz) * (z - qz); //calc distance//

					if (dist <= searchRad * searchRad)
					{
						int vx = FragVertex[findex];

						NbVertex[qoffset + pcount] = vx;
						NbVertexDist[qoffset + pcount] = GenerateVertexDistKey(q, dist);

						pcount++;
					}
						

				}


			}
		}


		q += gridDim.x * blockDim.x;
	}

}

void FillDistanceCuda(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragX, float* FragY, float* FragZ, int* FragVertex, int* qnoffset, int* NbVertex, unsigned long long* NbVertexDist)
{
	int gridsize = qnum / blocksize + 1;
	if (gridsize > maxblocks) gridsize = maxblocks;

	FillDistanceKernel << < gridsize, blocksize >> > (qnum, len, searchRad, globalW, globalH, pvmMat, xfcount, xfoffset, vpos, FragX, FragY, FragZ, FragVertex, qnoffset, NbVertex, NbVertexDist);
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

void CopyKNeighborsCuda(int k, float SearchRad, int qnum, int len, int* qncount, int NbsNum, int* NbVertex, int vnum, float* vpos, vector<vector<int>>& Nbs)
{
	int* NbVertexHost = new int[NbsNum];
	cudaMemcpy(NbVertexHost, NbVertex, NbsNum * sizeof(int), cudaMemcpyDeviceToHost);
	int* qncountHost = new int[qnum];
	cudaMemcpy(qncountHost, qncount, qnum * sizeof(int), cudaMemcpyDeviceToHost);
	float* vposHost = new float[vnum * 3];
	cudaMemcpy(vposHost, vpos, vnum * 3 * sizeof(float), cudaMemcpyDeviceToHost);

	int offset = 0;
	for (int q = 0; q < qnum; q++)
	{

		int copyn;
		if (qncountHost[q] < k)
			copyn = qncountHost[q];
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
		offset += qncountHost[q];

	}

}
