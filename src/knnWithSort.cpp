#include "DDS.h"
#include<nvml.h>

void DDS::kNNsearchWithSort(int k, float SearchRad)
{
	float milliseconds;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);


	qnum = vxPos->size();
	//qnum = 10;
	//qnum = 1000;

	float cellWidth = viewWidth / globalW;

	//int scrRad = 2;
	int scrRad = SearchRad / cellWidth;
	len = 2 * scrRad + 1;

	cudaMalloc((void**)&pixelIn, len * len * sizeof(bool));
	//FillAllWithValue(pixelIn, len * len, false);
	FillAllWithValue(pixelIn, len * len, true); //all true, temporarily !!

	//make a count array of size sqr * qs
	cudaMalloc((void**)&sncount, len * len * qnum * sizeof(int));
	FillAllWithValue(sncount, len * len * qnum, 0);

	//thread per(q, pixel): copy counts(0 for out pixels)
	CopyCountsCudaS(qnum, len, SearchRad, cellWidth, globalW, globalH, matrixVM, matrixPVM, vpos, xfcount, xfoffset, FragDepth, pixelIn, sncount);


	//make offset array(not sure how to do. call thrust multi times ? )
	cudaMalloc((void**)&snoffset, len * len * qnum * sizeof(int));
	CreateNbsOffsetArrayCuda(len * len * qnum, sncount, snoffset);

	//get sums in sums array(of size q)
	NbsNum = SumNbsCuda(qnum * len * len, sncount);

	// make vertex and distance arrays(of size sum of sums)
	cudaMalloc((void**)&NbVertexDist, NbsNum * sizeof(unsigned long long));
	cudaMalloc((void**)&NbVertex, NbsNum * sizeof(int));

	FillAllWithValue(sncount, len * len * qnum, 0);

	//thread per(q, pixel) : calculate distance to q, save vertex id, save(q, distance)
	FillDistanceCudaS(qnum, len, SearchRad, cellWidth, globalW, globalH, matrixVM, matrixPVM, vpos, xfcount, xfoffset, FragVertex, FragDepth, pixelIn, sncount, snoffset, NbVertex, NbVertexDist);

	//sort the big array!
	SortNeighborsCuda(NbsNum, NbVertex, NbVertexDist);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);


	//copy in new array
	Nbs.resize(qnum); for (int i = 0; i < qnum; i++) { Nbs[i].resize(k); for (int j = 0; j < k; j++) Nbs[i][j] = -1; }
	CopyKNeighborsCuda(k, SearchRad, qnum, len, sncount, NbsNum, NbVertex, vxPos->size(), vpos, Nbs);

	/*nvmlDevice_t dev;
	nvmlDeviceGetHandleByIndex(0, &dev);
	nvmlMemory_t mem;
	nvmlReturn_t nvmlret = nvmlDeviceGetMemoryInfo(dev, &mem);*/

	//size_t freeM, totalM;
	//cudaMemGetInfo(&freeM, &totalM);

	cout << "num of all nbs for all qs: " << NbsNum << "\n";

	cout << "***kNN search time: " << milliseconds << '\n';

	//for testing purposes
	for (int q = 0; q < 10; q++)
	{
		cout << "k neareset neighbors of " << q << " are: ";
		for (int j = 0; j < Nbs[q].size(); j++)
		{
			cout << Nbs[q][j] << " ";
		}
		cout << "\n";

	}

}

