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

__global__
void CalculateSquareSizeKernel(int qnum, float* qrads, bool* qkfound, float cellWidth, int* qscount) //compute len, and save sqr for each q//
{
	int q = blockIdx.x * blockDim.x + threadIdx.x;
	if (q < qnum)
	{
		if (!qkfound[q])
		{
			float rad = qrads[q];

			int scrRad = rad / cellWidth;
			int len = 2 * scrRad + 1;

			qscount[q] = len * len;
		}
	}
}

void CalculateSquareSizeCuda(int qnum, float* qrads, bool* qkfound, float cellWidth, int* qscount)
{
	CalculateSquareSizeKernel << < qnum / 256 + 1, 256 >> > (qnum, qrads, qkfound, cellWidth, qscount);
}

void CreateSquaresOffsetArrayCuda(int n, int* qscount, int* qsoffset)
{

	thrust::device_ptr<int> o = thrust::device_pointer_cast(qsoffset);
	thrust::device_ptr<int> c = thrust::device_pointer_cast(qscount);

	//call thrust function
	thrust::exclusive_scan(c, c + n, o);

}


int SumSPixelsCuda(int n, int* qscount)
{
	thrust::device_ptr<int> c = thrust::device_pointer_cast(qscount);

	//get count of qscount//
	int sPixelsNum = thrust::reduce(c, c + n, (int)0, thrust::plus<int>());


	return sPixelsNum;
}

/*__global__
void DebugBinaryKernel(int qnum, int sPixelsNum, int* qscount, int* qsoffset, bool* qspxlBool)
{
	int qspxl = blockIdx.x * blockDim.x + threadIdx.x;


	if (qspxl < sPixelsNum)
	{
		int q;

		if (qspxl >= qsoffset[qnum - 1])
			qspxlBool[qspxl] = true;
		else
		{
			//use binary search to find the q from qsoffset
			int first = 0; int last = qnum - 1;
			int mid;
			bool found = false;
			while (!found && first <= last)
			{
				mid = (first + last) / 2;
				if (qspxl >= qsoffset[mid - 1] && qspxl < qsoffset[mid])
					found = true;
				else
				{
					if (qspxl < qsoffset[mid])
						last = mid - 1;
					else
						first = mid + 1;
				}
			}

			if (found)
				qspxlBool[qspxl] = true;
		}
	}
}
void DebugBinaryCuda(int qnum, int sPixelsNum, int* qscount, int* qsoffset, bool* qspxlBool)
{
	DebugBinaryKernel << < sPixelsNum / 256 + 1, 256 >> > (qnum, sPixelsNum, qscount, qsoffset, qspxlBool);
}*/

__global__
void CountNeighborsKernel(int qnum, int sPixelsNum, int* qscount, int* qsoffset, bool* qkfound, int globalW, int globalH, float* pvmMat, float* vpos, float* qrads, int* xfcount, int* xfoffset, int* FragVertex, int* qncount)
{
	//for each q/, go through
	//project q, get the pixel
	//go through all the points in this pixel
	//start from beginning, add as long as distance lass than rad
	//

	int qspxl = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (qspxl < sPixelsNum)
	{
		int q;

		if (qspxl >= qsoffset[qnum - 1])
			q = qnum - 1;
		else
		{
			//use binary search to find the q from qsoffset
			int first = 0; int last = qnum - 1;
			int mid;
			bool found = false;
			//while (!found && first <= last)
			while (!found)
			{
				mid = (first + last) / 2;
				if (qspxl >= qsoffset[mid - 1] && qspxl < qsoffset[mid])
					found = true;
				else
				{
					if (qspxl < qsoffset[mid])
						last = mid - 1;
					else
						first = mid + 1;
				}
			}


			//if (!found) return;

			q = mid - 1;
		}

		//q = 5;

		if (!qkfound[q])
		{
			int qsnum = qscount[q];
			int len = sqrtf(qsnum);

			float rad = qrads[q];


			int spxl = qspxl - qsoffset[q]; //check edge cases !!!
			int sx = spxl % len;
			int sy = spxl / len;

			//get vertex//
			float qx = vpos[3 * q + 0];
			float qy = vpos[3 * q + 1];
			float qz = vpos[3 * q + 2];
			float qw = 1.0;

			float qposXpvm = pvmMat[0] * qx + pvmMat[4] * qy + pvmMat[8] * qz + pvmMat[12] * qw;
			float qposYpvm = pvmMat[1] * qx + pvmMat[5] * qy + pvmMat[9] * qz + pvmMat[13] * qw;
			float qposZpvm = pvmMat[2] * qx + pvmMat[6] * qy + pvmMat[10] * qz + pvmMat[14] * qw;
			float qposWpvm = pvmMat[3] * qx + pvmMat[7] * qy + pvmMat[11] * qz + pvmMat[15] * qw;

			//exact pixel of q//
			int qxscreen = (int)(((qposXpvm / qposWpvm) / 2 + 0.5) * globalW);
			int qyscreen = (int)(((qposYpvm / qposWpvm) / 2 + 0.5) * globalH);

			//pixel in the square, based on thread//
			int xscreen = (qxscreen - len / 2) + sx;
			int yscreen = (qyscreen - len / 2) + sy;
			int pxl = xscreen + yscreen * globalW;

			if (xscreen<0 || xscreen>globalW - 1 || yscreen<0 || yscreen>globalH - 1)
				return;

			//decide what is the distance range allowed in the cetain pixel (wrt the qpixel)

			//go through distances of vxs in this pixel
			int pcount = 0;
			for (int j = 0; j < xfcount[pxl]; j++)
			{
				int pindex = xfoffset[pxl] + j;
				int p = FragVertex[pindex];

				float x = vpos[3 * p + 0];
				float y = vpos[3 * p + 1];
				float z = vpos[3 * p + 2];

				float qpdst = (x - qx) * (x - qx) + (y - qy) * (y - qy) + (z - qz) * (z - qz);

				if (qpdst < rad * rad)
					pcount++;

			}

			//accumulate counts//
			atomicAdd(&qncount[q], pcount);
		}
	}


}

void CountNeighborsCuda(int qnum, int sPixelsNum, int* qscount, int* qsoffset, bool* qkfound, int globalW, int globalH, float* pvmMat, float* vpos, float* qrads, int* xfcount, int* xfoffset, int* FragVertex, int* qncount)
{
	CountNeighborsKernel << < sPixelsNum / 256 + 1, 256 >> > (qnum, sPixelsNum, qscount, qsoffset, qkfound, globalW, globalH, pvmMat, vpos, qrads, xfcount, xfoffset, FragVertex, qncount);
}

void CreateNbsOffsetArrayCudaS(int n, int* qncount, int* qnoffset)
{

	thrust::device_ptr<int> o = thrust::device_pointer_cast(qnoffset);
	thrust::device_ptr<int> c = thrust::device_pointer_cast(qncount);

	//call thrust function
	thrust::exclusive_scan(c, c + n, o);
}

int SumNbsCudaS(int n, int* qncount)
{

	thrust::device_ptr<int> c = thrust::device_pointer_cast(qncount);

	//get count of xfcount//
	int NbsNum = thrust::reduce(c, c + n, (int)0, thrust::plus<int>());

	return NbsNum;
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
void FillDistanceKernelS(int qnum, int sPixelsNum, int* qscount, int* qsoffset, bool* qkfound, int globalW, int globalH, float* pvmMat, float* vpos, float* qrads, int* xfcount, int* xfoffset, int* FragVertex, int* qncount, int* qnoffset, int* NbVertex, unsigned long long* NbVertexDist)
{

	int qspxl = blockIdx.x * blockDim.x + threadIdx.x;


	if (qspxl < sPixelsNum)
	{
		int q;

		if (qspxl >= qsoffset[qnum - 1])
			q = qnum - 1;
		else
		{
			//use binary search to find the q from qsoffset
			int first = 0; int last = qnum - 1;
			int mid;
			bool found = false;
			//while (!found && first <= last)
			while (!found)
			{
				mid = (first + last) / 2;
				if (qspxl >= qsoffset[mid - 1] && qspxl < qsoffset[mid])
					found = true;
				else
				{
					if (qspxl < qsoffset[mid])
						last = mid - 1;
					else
						first = mid + 1;
				}
			}


			//if (!found) return;

			q = mid - 1;
		}

		//q = 5;

		int qsnum = qscount[q];
		int len = sqrtf(qsnum);

		float rad = qrads[q];

		int offset = qnoffset[q];


		int spxl = qspxl - qsoffset[q]; //check edge cases !!!
		int sx = spxl % len;
		int sy = spxl / len;

		//get vertex//
		float qx = vpos[3 * q + 0];
		float qy = vpos[3 * q + 1];
		float qz = vpos[3 * q + 2];
		float qw = 1.0;

		float qposXpvm = pvmMat[0] * qx + pvmMat[4] * qy + pvmMat[8] * qz + pvmMat[12] * qw;
		float qposYpvm = pvmMat[1] * qx + pvmMat[5] * qy + pvmMat[9] * qz + pvmMat[13] * qw;
		float qposZpvm = pvmMat[2] * qx + pvmMat[6] * qy + pvmMat[10] * qz + pvmMat[14] * qw;
		float qposWpvm = pvmMat[3] * qx + pvmMat[7] * qy + pvmMat[11] * qz + pvmMat[15] * qw;

		//exact pixel of q//
		int qxscreen = (int)(((qposXpvm / qposWpvm) / 2 + 0.5) * globalW);
		int qyscreen = (int)(((qposYpvm / qposWpvm) / 2 + 0.5) * globalH);

		//pixel in the square, based on thread//
		int xscreen = (qxscreen - len / 2) + sx;
		int yscreen = (qyscreen - len / 2) + sy;
		int pxl = xscreen + yscreen * globalW;

		if (xscreen<0 || xscreen>globalW - 1 || yscreen<0 || yscreen>globalH - 1)
			return;

		//decide what is the distance range allowed in the cetain pixel (wrt the qpixel)

		//go through distances of vxs in this pixel
		int pos;
		for (int j = 0; j < xfcount[pxl]; j++)
		{
			int pindex = xfoffset[pxl] + j;
			int p = FragVertex[pindex];

			float x = vpos[3 * p + 0];
			float y = vpos[3 * p + 1];
			float z = vpos[3 * p + 2];

			float qpdst = (x - qx) * (x - qx) + (y - qy) * (y - qy) + (z - qz) * (z - qz);

			if (qpdst < rad * rad)
			{

				pos = atomicAdd(&qncount[q], 1);

				NbVertexDist[offset + pos] = GenerateVertexDistKeyS(q, qpdst);
				NbVertex[offset + pos] = p;

			}



		}

	}


}

void FillDistanceCudaS(int qnum, int sPixelsNum, int* qscount, int* qsoffset, bool* qkfound, int globalW, int globalH, float* pvmMat, float* vpos, float* qrads, int* xfcount, int* xfoffset, int* FragVertex, int* qncount, int* qnoffset, int* NbVertex, unsigned long long* NbVertexDist)
{
	FillDistanceKernelS << < sPixelsNum / 256 + 1, 256 >> > (qnum, sPixelsNum, qscount, qsoffset, qkfound, globalW, globalH, pvmMat, vpos, qrads, xfcount, xfoffset, FragVertex, qncount, qnoffset, NbVertex, NbVertexDist);
}


__global__
void UpdateRadsKernel(int qnum, int k, int* qncount, bool* qkfound, float* qrads)
{
	int q = blockIdx.x * blockDim.x + threadIdx.x;

	if (q < qnum)
	{
		
		//for testing purposes
		qkfound[q] = true;
		//if (q == 500000)
		//	qkfound[q] = false;

		/*int qnbnum = qncount[q];
		if (qnbnum >= k && qnbnum <= k + 3)
			qkfound[q] = true;
		else
		{
			//update qrads//
		}*/

	}
}

void UpdateRadsCuda(int qnum, int k, int* qncount, bool* qkfound, float* qrads) //check count. mainly search for any value less than k or more than k+3//
{
	UpdateRadsKernel << <qnum / 256 + 1, 256 >> > (qnum, k, qncount, qkfound, qrads);
}

bool AllKNbsFoundCuda(int qnum, bool* qkfound)
{
	thrust::device_ptr<bool> a = thrust::device_pointer_cast(qkfound);
	thrust::device_vector<bool>::iterator iter = thrust::find(thrust::device, a, a + qnum, false);
	
	return (&iter[0] == a + qnum);

}



void SortNeighborsCudaS(int NbsNum, int* NbVertex, unsigned long long* NbVertexDist)
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

void CopyKNeighborsCudaS(int qnum, int* qncount, int NbsNum, int* NbVertex, vector<vector<int>>& Nbs)
{
	int* NbVertexHost = new int[NbsNum];
	cudaMemcpy(NbVertexHost, NbVertex, NbsNum * sizeof(int), cudaMemcpyDeviceToHost);
	int* qncountHost = new int[qnum];
	cudaMemcpy(qncountHost, qncount, qnum * sizeof(int), cudaMemcpyDeviceToHost);

	int offset = 0;
	for (int q = 0; q < qnum; q++)
	{
		
		int copyn = qncountHost[q];
		
		int counter = 0;
		for (int i = 0; i < copyn; i++)
		{
			if (i > 9) continue;
			int vx = NbVertexHost[offset + i];
			Nbs[q][counter++] = vx;
		}
		offset += copyn;

	}

}


