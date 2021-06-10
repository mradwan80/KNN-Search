#include "DDS.h"
#include<nvml.h>

void DDS::TestFindKNN()
{
	//project 0. find pixel
	//get counts of pixel, pixel+1,pixel-1, pixel+w,pixel-w
	//get first 25 counts in xncount

	int q = 0;
	float x = (*vxPos)[q].x;
	float y = (*vxPos)[q].y;
	float z = (*vxPos)[q].z;
	float w = 1.0;

	/*float posXpvm = pvmMat[0] * x + pvmMat[4] * y + pvmMat[8] * z + pvmMat[12] * w;
	float posYpvm = matrixPVM[1] * x + matrixPVM[5] * y + matrixPVM[9] * z + matrixPVM[13] * w;
	float posZpvm = matrixPVM[2] * x + matrixPVM[6] * y + matrixPVM[10] * z + matrixPVM[14] * w;
	float posWpvm = matrixPVM[3] * x + matrixPVM[7] * y + matrixPVM[11] * z + matrixPVM[15] * w;*/

	glm::vec4 pos = glm::vec4((*vxPos)[q].x, (*vxPos)[q].y, (*vxPos)[q].z, 1.0);

	glm::vec4 pospvm = pvmMat * pos;


	//exact pixel of q//
	int qxscreen = (int)(((pospvm.x / pospvm.w) / 2 + 0.5) * globalW);
	int qyscreen = (int)(((pospvm.y / pospvm.w) / 2 + 0.5) * globalH);
	int qpxl = qxscreen + qyscreen * globalW;

	int* xfcountHost = new int[globalW * globalH];
	cudaMemcpy(xfcountHost, xfcount, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);

	int cnt0 = xfcountHost[qpxl];
	int cnt1 = xfcountHost[qpxl - 1];
	int cnt2 = xfcountHost[qpxl + 1];
	int cnt3 = xfcountHost[qpxl - globalW];
	int cnt4 = xfcountHost[qpxl + globalW];


	int* xfoffsetHost = new int[globalW * globalH];
	cudaMemcpy(xfoffsetHost, xfoffset, globalW * globalH * sizeof(int), cudaMemcpyDeviceToHost);
	
	int* FragVertexHost = new int[FragsNum];
	cudaMemcpy(FragVertexHost, FragVertex, FragsNum * sizeof(int), cudaMemcpyDeviceToHost);
		

	int* sncountHost = new int[qnum * len * len];
	cudaMemcpy(sncountHost, sncount, qnum * len * len * sizeof(int), cudaMemcpyDeviceToHost);

	int* snoffsetHost = new int[qnum * len * len];
	cudaMemcpy(snoffsetHost, snoffset, qnum * len * len * sizeof(int), cudaMemcpyDeviceToHost);

	int* NbVertexHost = new int[NbsNum];
	cudaMemcpy(NbVertexHost, NbVertex, NbsNum * sizeof(int), cudaMemcpyDeviceToHost);


	int m = 3;
}


void DDS::kNNsearch(int k, float SearchRad)
{
	float milliseconds;

	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	

	cudaEvent_t nosortstop;
	cudaEventCreate(&nosortstop);
	cudaEvent_t auxstart, auxstop;
	cudaEventCreate(&auxstart);
	cudaEventCreate(&auxstop);

	

	////////////
	//begin//
	////////////

	cudaEventRecord(start);

	qnum = vxPos->size();
	//qnum = 10;
	//qnum = 1000;

	float cellWidth = viewWidth / globalW;

	//int scrRad = 2;
	int scrRad = SearchRad / cellWidth;
	len = 2 * scrRad + 1;
	//len = 1;

	cout << "len = " << len << "\n";
	
	////////////
	//copy counts//
	////////////

	cudaEventRecord(auxstart);

	//make a count array of size sqr * qs
	cudaMalloc((void**)&sncount, len * len * qnum * sizeof(int));
	FillAllWithValue(sncount, len * len * qnum, 0);

	//thread per(q, pixel): copy counts(0 for out pixels)
	CopyCountsCuda(qnum, len, SearchRad, globalW, globalH, matrixPVM, vpos, xfcount, xfoffset, FragVertex, sncount);


	cudaEventRecord(auxstop);
	cudaEventSynchronize(auxstop);
	cudaEventElapsedTime(&milliseconds, auxstart, auxstop);
	float countTime = milliseconds;

	////////////
	//create nbs offset//
	////////////

	cudaEventRecord(auxstart);

	//make offset array(not sure how to do. call thrust multi times ? )
	cudaMalloc((void**)&snoffset, len * len * qnum * sizeof(int));
	CreateNbsOffsetArrayCuda(len * len * qnum, sncount, snoffset);

	//get sums in sums array(of size q)
	NbsNum = SumNbsCuda(qnum * len * len, sncount);
	//cout << "NbsNum=" << NbsNum << "\n";

	// make vertex and distance arrays(of size sum of sums)
	cudaMalloc((void**)&NbVertexDist, NbsNum * sizeof(unsigned long long));
	cudaMalloc((void**)&NbVertex, NbsNum * sizeof(int));
	
	FillAllWithValue(sncount, len * len * qnum, 0);

	cudaEventRecord(auxstop);
	cudaEventSynchronize(auxstop);
	cudaEventElapsedTime(&milliseconds, auxstart, auxstop);
	float offsetTime = milliseconds;


	////////////
	//fill distance//
	////////////

	cudaEventRecord(auxstart);

	//thread per(q, pixel) : calculate distance to q, save vertex id, save(q, distance)
	FillDistanceCuda(qnum, len, SearchRad, globalW, globalH, matrixPVM, vpos, xfcount, xfoffset, FragVertex, sncount, snoffset, NbVertex, NbVertexDist);
	

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

	cudaEventRecord(auxstop);
	cudaEventSynchronize(auxstop);
	cudaEventElapsedTime(&milliseconds, auxstart, auxstop);
	float sortTime = milliseconds;


	////////////
	//end//
	////////////

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	
	//copy in new array
	Nbs.resize(qnum); for (int i = 0; i < qnum; i++) {		Nbs[i].resize(k); for (int j = 0; j < k; j++) Nbs[i][j] = -1;	}
	CopyKNeighborsCuda(k, SearchRad, qnum, len, sncount, NbsNum, NbVertex, vxPos->size(), vpos, Nbs);
	
	//TestFindKNN();

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
	cout << "***kNN without sort time: " << NoSortTotalTime << '\n';

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

/*void DDS::kNNsearchSimple(int k, float SearchRad)
{
	float milliseconds;


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	

	qnum = vxPos->size();
	//qnum = 10;
	//qnum = 1000;

	float cellWidth = viewWidth / globalW;

	//int scrRad = 2;
	int scrRad = SearchRad / cellWidth;
	len = 2 * scrRad + 1;

	cout << "len = " << len << "\n";

	cudaEventRecord(start);
	
	////////////
	//copy counts//
	////////////

	//make a count array of size sqr * qs
	cudaMalloc((void**)&qncount, qnum * sizeof(int));
	FillAllWithValue(qncount, qnum, 0);

	//thread per(q, pixel): copy counts(0 for out pixels)
	CopyCountsSimpleCuda(qnum, len, SearchRad, globalW, globalH, matrixPVM, vpos, xfcount, xfoffset, FragVertex, qncount);


	
	
	////////////
	//end//
	////////////

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);


	//copy in new array
	//Nbs.resize(qnum); for (int i = 0; i < qnum; i++) { Nbs[i].resize(k); for (int j = 0; j < k; j++) Nbs[i][j] = -1; }
	//CopyKNeighborsCuda(k, SearchRad, qnum, len, sncount, NbsNum, NbVertex, vxPos->size(), vpos, Nbs);


	cout << "num of all nbs for all qs: " << NbsNum << "\n";


	cout << "***kNN search time: " << milliseconds << '\n';

	
	NbsNum = SumNbsCuda(qnum, qncount);
	cout << "NbsNum=" << NbsNum << "\n";


}*/

void DDS::PrepareCoordsSimple()
{
	cudaMalloc((void**)&FragX, FragsNum * sizeof(float));
	cudaMalloc((void**)&FragY, FragsNum * sizeof(float));
	cudaMalloc((void**)&FragZ, FragsNum * sizeof(float));

	PrepareCoordsSimpleCuda(FragsNum, FragVertex, vpos, FragX, FragY, FragZ);

}

void DDS::kNNsearchSimple(int k, float SearchRad)
{
	PrepareCoordsSimple();
	//cudaDeviceSynchronize();

	float milliseconds;


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);




	qnum = vxPos->size();
	//qnum = 10;
	//qnum = 1000;

	float cellWidth = viewWidth / globalW;

	//int scrRad = 2;
	int scrRad = SearchRad / cellWidth;
	len = 2 * scrRad + 1;
	//len = 1;

	cout << "len = " << len << "\n";

	cudaEventRecord(start);

	////////////
	//copy counts//
	////////////

	//make a count array of size sqr * qs
	cudaMalloc((void**)&qncount, qnum * sizeof(int));
	FillAllWithValue(qncount, qnum, 0);

	//thread per(q, pixel): copy counts(0 for out pixels)
	CopyCountsSimpleCuda(qnum, len, SearchRad, globalW, globalH, matrixPVM, xfcount, xfoffset, vpos, FragX, FragY, FragZ, qncount);




	////////////
	//end//
	////////////

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);


	//copy in new array
	//Nbs.resize(qnum); for (int i = 0; i < qnum; i++) { Nbs[i].resize(k); for (int j = 0; j < k; j++) Nbs[i][j] = -1; }
	//CopyKNeighborsCuda(k, SearchRad, qnum, len, sncount, NbsNum, NbVertex, vxPos->size(), vpos, Nbs);


	cout << "num of all nbs for all qs: " << NbsNum << "\n";


	cout << "***kNN search time: " << milliseconds << '\n';


	NbsNum = SumNbsCuda(qnum, qncount);
	cout << "NbsNum=" << NbsNum << "\n";


}

