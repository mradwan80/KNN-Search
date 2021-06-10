#include "DDS.h"
#include<nvml.h>

void DDS::kNNsearchWithSort(int k, float SearchRad)
{
	GetFragsCoords();

	float milliseconds;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEvent_t nosortstop;
	cudaEventCreate(&nosortstop);

	cudaEvent_t auxstart, auxstop;
	cudaEventCreate(&auxstart);
	cudaEventCreate(&auxstop);

	
	qnum = vxPos->size();
	//qnum = 10;
	//qnum = 1000;

	float cellWidth = viewWidth / globalW;

	//int scrRad = 2;
	int scrRad = SearchRad / cellWidth;
	len = 2 * scrRad + 1;

	////////////
	//begin//
	////////////

	cudaEventRecord(start);

	////////////
	//copy counts//
	////////////

	cudaEventRecord(auxstart);

	//make a count array//
	cudaMalloc((void**)&qncount, qnum * sizeof(int));

	CopyCountsCudaS(qnum, len, SearchRad, cellWidth, globalW, globalH, matrixVM, matrixPVM, xfcount, xfoffset, vpos, FragDepth, qncount);

	checkCUDAError("Counting candidate neighbors failed");

	cudaEventRecord(auxstop);
	cudaEventSynchronize(auxstop);
	cudaEventElapsedTime(&milliseconds, auxstart, auxstop);
	float countTime = milliseconds;

	////////////
	//create nbs offset//
	////////////

	cudaEventRecord(auxstart);

	//make offset array//
	cudaMalloc((void**)&qnoffset, qnum * sizeof(int));
	CreateNbsOffsetArrayCuda(qnum, qncount, qnoffset);

	//get sums in sums array(of size q)
	NbsNum = SumNbsCuda(qnum, qncount);

	// make vertex and distance arrays(of size sum of sums)//
	cudaMalloc((void**)&NbVertexDist, NbsNum * sizeof(unsigned long long));
	cudaMalloc((void**)&NbVertex, NbsNum * sizeof(int));

	checkCUDAError("Creating auxiliary vectors failed");

	cudaEventRecord(auxstop);
	cudaEventSynchronize(auxstop);
	cudaEventElapsedTime(&milliseconds, auxstart, auxstop);
	float offsetTime = milliseconds;

	////////////
	//fill distance//
	////////////

	cudaEventRecord(auxstart);

	FillDistanceCudaS(qnum, len, SearchRad, cellWidth, globalW, globalH, matrixVM, matrixPVM, xfcount, xfoffset, vpos, FragX, FragY, FragZ, FragVertex, FragDepth, qnoffset, NbVertex, NbVertexDist);

	checkCUDAError("Getting neighbors distances failed");

	cudaEventRecord(auxstop);
	cudaEventSynchronize(auxstop);
	cudaEventElapsedTime(&milliseconds, auxstart, auxstop);
	float distTime = milliseconds;

	cudaEventRecord(nosortstop);
	cudaEventSynchronize(nosortstop);
	cudaEventElapsedTime(&milliseconds, start, nosortstop);
	float NoSortTotalTime = milliseconds;

	////////////
	//sort//
	////////////

	cudaEventRecord(auxstart);

	//sort the big array!
	SortNeighborsCuda(NbsNum, NbVertex, NbVertexDist);

	checkCUDAError("Sorting neighbors failed");

	cudaEventRecord(auxstop);
	cudaEventSynchronize(auxstop);
	cudaEventElapsedTime(&milliseconds, auxstart, auxstop);
	float sortTime = milliseconds;


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	////////////
	//end//
	////////////

	//copy in new array//
	Nbs.resize(qnum); for (int i = 0; i < qnum; i++) { Nbs[i].resize(k); for (int j = 0; j < k; j++) Nbs[i][j] = -1; }
	CopyKNeighborsCuda(k, SearchRad, qnum, len, qncount, NbsNum, NbVertex, vxPos->size(), vpos, Nbs);

	/*nvmlDevice_t dev;
	nvmlDeviceGetHandleByIndex(0, &dev);
	nvmlMemory_t mem;
	nvmlReturn_t nvmlret = nvmlDeviceGetMemoryInfo(dev, &mem);*/

	//size_t freeM, totalM;
	//cudaMemGetInfo(&freeM, &totalM);

	cout << "num of all nbs for all qs: " << NbsNum << "\n";
	cout << "count time: " << countTime << "\n";
	cout << "offset vector time: " << offsetTime << "\n";
	cout << "filling distance time: " << distTime << "\n";
	cout << "sort time: " << sortTime << "\n";
	cout << "***kNN search time: " << milliseconds << '\n';
	cout << "***kNN (without sorting output) search time: " << NoSortTotalTime << '\n';

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

