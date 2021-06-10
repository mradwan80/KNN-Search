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

__global__ void CopyCountsKernel(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, float* vpos, int* xfcount, int* xfoffset, int* FragVertex, int* sncount)
{
	int qspxl = blockIdx.x * blockDim.x + threadIdx.x;

	while (qspxl < qnum * len * len)
	{
		int q = qspxl / (len * len);
		int spxl = qspxl % (len * len);
		int sx = spxl % len;
		int sy = spxl / len;

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

		int offset = xfoffset[pxl];
		int pcount = 0;
		for (int f = 0; f < xfcount[pxl]; f++)
		{
			int v = FragVertex[f + offset]; //get vertex//

			//get pos
			float x = vpos[3 * v + 0];
			float y = vpos[3 * v + 1];
			float z = vpos[3 * v + 2];
			float w = 1.0;

			float dist = (x - qx) * (x - qx) + (y - qy) * (y - qy) + (z - qz) * (z - qz); //calc distance//

			if (dist <= searchRad * searchRad)
				pcount++;

		}

		sncount[qspxl] = pcount;

		qspxl += gridDim.x * blockDim.x;
	}

}

void CopyCountsCuda(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, float* vpos, int* xfcount, int* xfoffset, int* FragVertex, int* sncount)
{
	int gridsize = (qnum * len * len) / 512 + 1;
	if (gridsize > 65535) gridsize = 65535;

	CopyCountsKernel << < gridsize, 512 >> > (qnum, len, searchRad, globalW, globalH, pvmMat, vpos, xfcount, xfoffset, FragVertex, sncount);

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
void FillDistanceKernel(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, float* vpos, int* xfcount, int* xfoffset, int* FragVertex, int* sncount, int* snoffset, int* NbVertex, unsigned long long* NbVertexDist)
{
	int qspxl = blockIdx.x * blockDim.x + threadIdx.x;

	while (qspxl < qnum * len * len)
	{
		int q = qspxl / (len * len);
		int spxl = qspxl % (len * len);
		int sx = spxl % len;
		int sy = spxl / len;

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
		int xscreen = (qxscreen - len/2) + sx;
		int yscreen = (qyscreen - len/2) + sy;
		int pxl = xscreen + yscreen * globalW;

		if (xscreen<0 || xscreen>globalW - 1 || yscreen<0 || yscreen>globalH - 1)
			return;

		if (xfcount[pxl] == 0)
			return;

		int offset = xfoffset[pxl];
		for (int v = 0; v < xfcount[pxl]; v++)
		{
			int vx = FragVertex[v + offset];
			float x = vpos[3 * vx + 0];
			float y = vpos[3 * vx + 1];
			float z = vpos[3 * vx + 2];

			float dist = (x - qx) * (x - qx) + (y - qy) * (y - qy) + (z - qz) * (z - qz);

			if (dist <= searchRad * searchRad)
			{
				int pos = atomicAdd(&sncount[qspxl], 1);

				NbVertex[snoffset[qspxl] + pos] = vx;
				NbVertexDist[snoffset[qspxl] + pos] = GenerateVertexDistKey(q, dist);
			}
			


		}


		qspxl += gridDim.x * blockDim.x;
	}

}

void FillDistanceCuda(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, float* vpos, int* xfcount, int* xfoffset, int* FragVertex, int* sncount, int* snoffset, int* NbVertex, unsigned long long* NbVertexDist)
{

	int gridsize = (qnum * len * len) / 512 + 1;
	if (gridsize > 65535) gridsize = 65535;

	FillDistanceKernel << < gridsize, 512 >> > (qnum, len, searchRad, globalW, globalH, pvmMat, vpos, xfcount, xfoffset, FragVertex, sncount, snoffset, NbVertex, NbVertexDist);


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


///////////////////////////////
///////////////////////////////
///////////////////////////////
///////////////////////////////
///////////////////////////////

__global__ void PrepareCoordsSimpleKernel(int N, int* FragVertex, float* vpos, float* FragX, float* FragY, float* FragZ)
{
	int frag = blockIdx.x * blockDim.x + threadIdx.x;
	if (frag < N)
	{
		int v = FragVertex[frag];
		FragX[frag] = vpos[3 * v + 0];
		FragY[frag] = vpos[3 * v + 1];
		FragZ[frag] = vpos[3 * v + 2];
	}
}

void PrepareCoordsSimpleCuda(int N, int* FragVertex, float* vpos, float* FragX, float* FragY, float* FragZ)
{
	PrepareCoordsSimpleKernel << <N / 512 + 1, 512 >> > (N, FragVertex, vpos, FragX, FragY, FragZ);
}


/*__global__ void CopyCountsSimpleKernel(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragX, float* FragY, float* FragZ, int* qncount)
{


	extern __shared__ int counts[];

	__shared__ int q, sx, sy;
	q = blockIdx.x;
	sx = blockIdx.y;
	sy = blockIdx.z;

	__shared__ float qx, qy, qz;
	__shared__ int xscreen, yscreen, pxl;
	__shared__ float searchRadSqr;
	__shared__ int count, offset;

	while (q < qnum)
	{
		if (threadIdx.x == 0)
		{
			//get vertex//
			qx = vpos[3 * q + 0];
			qy = vpos[3 * q + 1];
			qz = vpos[3 * q + 2];

			float posXpvm = pvmMat[0] * qx + pvmMat[4] * qy + pvmMat[8] * qz + pvmMat[12];
			float posYpvm = pvmMat[1] * qx + pvmMat[5] * qy + pvmMat[9] * qz + pvmMat[13];
			float posZpvm = pvmMat[2] * qx + pvmMat[6] * qy + pvmMat[10] * qz + pvmMat[14];
			float posWpvm = pvmMat[3] * qx + pvmMat[7] * qy + pvmMat[11] * qz + pvmMat[15];

			//exact pixel of q//
			int qxscreen = (int)(((posXpvm / posWpvm) / 2 + 0.5) * globalW);
			int qyscreen = (int)(((posYpvm / posWpvm) / 2 + 0.5) * globalH);

			//pixel in the square, based on thread//
			xscreen = (qxscreen - len / 2) + sx;
			yscreen = (qyscreen - len / 2) + sy;
			pxl = xscreen + yscreen * globalW;

			searchRadSqr = searchRad * searchRad;

			count = xfcount[pxl];
			offset = xfoffset[pxl];

			if (xscreen < 0 || xscreen >= globalW || yscreen < 0 || yscreen >= globalH)
				count = 0;
		}



		__syncthreads();


		int pcount = 0;
		int v;
		float x, y, z, diffx, diffy, diffz, dist = 0;
		for (int f = 0; f + threadIdx.x < count; f += 32)
		{

			//get pos
			x = FragX[offset + f + threadIdx.x];
			y = FragY[offset + f + threadIdx.x];
			z = FragZ[offset + f + threadIdx.x];


			diffx = x - qx;
			diffy = y - qy;
			diffz = z - qz;

			dist = diffx * diffx + diffy * diffy + diffz * diffz; //calc distance//

			if (dist <= searchRadSqr)
				pcount++;

		}

		counts[threadIdx.x] = pcount;

		__syncthreads();

		if (threadIdx.x == 0)
		{
			int total = 0;
			for (int i = 0; i < 32; i++)
				total += counts[i];

			atomicAdd(&qncount[q], total); //do atomic add
		}

		q += gridDim.x;

		__syncthreads();
	}
}



void CopyCountsSimpleCuda(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragX, float* FragY, float* FragZ, int* qncount)
{
	int gridsize = qnum;
	if (gridsize > 65535) gridsize = 65535;

	CopyCountsSimpleKernel << < dim3(gridsize, len, len), 32, sizeof(int) * 32 >> > (qnum, len, searchRad, globalW, globalH, pvmMat, xfcount, xfoffset, vpos, FragX, FragY, FragZ, qncount);
}*/

__global__ void CopyCountsSimpleKernel(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragX, float* FragY, float* FragZ, int* qncount)
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
				
				for (int f = 0; f < xfcount[pxl]; f++)
				{

					//get pos
					float x = FragX[f + offset];
					float y = FragY[f + offset];
					float z = FragZ[f + offset];
					float w = 1.0;

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



void CopyCountsSimpleCuda(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragX, float* FragY, float* FragZ, int* qncount)
{
	int gridsize = qnum / 512 + 1;
	if (gridsize > 65535) gridsize = 65535;

	CopyCountsSimpleKernel << < gridsize, 512 >> > (qnum, len, searchRad, globalW, globalH, pvmMat, xfcount, xfoffset, vpos, FragX, FragY, FragZ, qncount);
}


__global__ void CopyCountsSimpleKernel_2(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragX, float* FragY, float* FragZ, int* qncount)
{


	extern __shared__ int counts[];

	__shared__ int q;
	q = blockIdx.x;


	
	__shared__ float searchRadSqr;
	__shared__ float qx, qy, qz;
	__shared__ float posXpvm, posYpvm, posZpvm, posWpvm;
	__shared__ int qxscreen, qyscreen;


	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
	{
		searchRadSqr = searchRad * searchRad;

		//get vertex//
		qx = vpos[3 * q + 0];
		qy = vpos[3 * q + 1];
		qz = vpos[3 * q + 2];

		posXpvm = pvmMat[0] * qx + pvmMat[4] * qy + pvmMat[8] * qz + pvmMat[12];
		posYpvm = pvmMat[1] * qx + pvmMat[5] * qy + pvmMat[9] * qz + pvmMat[13];
		posZpvm = pvmMat[2] * qx + pvmMat[6] * qy + pvmMat[10] * qz + pvmMat[14];
		posWpvm = pvmMat[3] * qx + pvmMat[7] * qy + pvmMat[11] * qz + pvmMat[15];

		//exact pixel of q//
		qxscreen = (int)(((posXpvm / posWpvm) / 2 + 0.5) * globalW);
		qyscreen = (int)(((posYpvm / posWpvm) / 2 + 0.5) * globalH);
	}

	__syncthreads();

	int sx = threadIdx.y;
	int sy = threadIdx.z;

	//pixel in the square, based on thread//
	int xscreen = (qxscreen - len / 2) + sx;
	int yscreen = (qyscreen - len / 2) + sy;
	int pxl = xscreen + yscreen * globalW;



	int count = xfcount[pxl];
	int offset = xfoffset[pxl];

	if (xscreen < 0 || xscreen >= globalW || yscreen < 0 || yscreen >= globalH)
		count = 0;


	__syncthreads();


	int pcount = 0;
	int v;
	float x, y, z, diffx, diffy, diffz, dist = 0;
	for (int f = 0; f + threadIdx.x < count; f += 32)
	{

		//get pos
		x = FragX[offset + f + threadIdx.x];
		y = FragY[offset + f + threadIdx.x];
		z = FragZ[offset + f + threadIdx.x];


		diffx = x - qx;
		diffy = y - qy;
		diffz = z - qz;

		dist = diffx * diffx + diffy * diffy + diffz * diffz; //calc distance//

		if (dist <= searchRadSqr)
			pcount++;

	}

	counts[(threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = pcount;

	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
	{
		int total = 0;
		for (int i = 0; i < blockDim.z * blockDim.y * blockDim.x; i++)
			total += counts[i];

		//atomicAdd(&qncount[q], total); //do atomic add
		qncount[q] = total;
	}


}

void CopyCountsSimpleCuda_2(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragX, float* FragY, float* FragZ, int* qncount)
{

	CopyCountsSimpleKernel_2 << < qnum, dim3(32, len, len), sizeof(int)* len* len * 32 >> > (qnum, len, searchRad, globalW, globalH, pvmMat, xfcount, xfoffset, vpos, FragX, FragY, FragZ, qncount);
}


