#include "DDS.h"
#include<nvml.h>

//more difficult than I though !
//you should change rad and len per q !

void DDS::findKNNwithSort(int k, float SearchRad)
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

	cudaMalloc((void**)&qrads, qnum * sizeof(int));
	FillAllWithValue(qrads, qnum, SearchRad); //initialization//

	cudaMalloc((void**)&qkfound, qnum * sizeof(bool));
	FillAllWithValue(qkfound, qnum, false); //initialization//

	cudaMalloc((void**)&qncount, qnum * sizeof(int));

	cudaMalloc((void**)&qscount, qnum * sizeof(int));
	cudaMalloc((void**)&qsoffset, qnum * sizeof(int));


	int sPixelsNum;
	bool AllKsFound = false;
	while (!AllKsFound)
	{

		FillAllWithValue(qncount, qnum, 0); //initialization//

		CalculateSquareSizeCuda(qnum, qrads, qkfound, cellWidth, qscount); //compute len, and save sqr for each q//
		
		CreateSquaresOffsetArrayCuda(qnum, qscount, qsoffset); //offset//

		sPixelsNum = SumSPixelsCuda(qnum, qscount); //accumulate all square pixels//


		/////////////
		//////////////
		//testing
		/*bool* qspxlBool;
		cudaMalloc((void**)&qspxlBool, sPixelsNum * sizeof(bool));
		FillAllWithValue(qspxlBool, sPixelsNum, false);

		DebugBinaryCuda(qnum, sPixelsNum, qscount, qsoffset, qspxlBool);

		//get bool array and qsoffset. find which qspxls fail. try to check why
		int* qsoffsetHost = new int[qnum];
		cudaMemcpy(qsoffsetHost, qsoffset, qnum * sizeof(int), cudaMemcpyDeviceToHost);
		bool* qspxlBoolHost = new bool[sPixelsNum];
		cudaMemcpy(qspxlBoolHost, qspxlBool, sPixelsNum * sizeof(bool), cudaMemcpyDeviceToHost);
		for (int qs = 0; qs < sPixelsNum; qs++)
		{ 
			if (!qspxlBoolHost[qs])
				qs = qs;
		}*/
		/////////////
		//////////////


		CountNeighborsCuda(qnum, sPixelsNum, qscount, qsoffset, qkfound, globalW, globalH, matrixVM, matrixPVM, vpos, qrads, cellWidth, xfcount, xfoffset, FragDepth, qncount); //count. per(q,pixel). use binary search.
		cudaDeviceSynchronize(); //sync//
		
		UpdateRadsCuda(qnum, k, qncount, qkfound, qrads); //check count. mainly search for any value less than k or more than k+3//
		
		AllKsFound = AllKNbsFoundCuda(qnum, qkfound); //update AllKsFound //

	}

	cudaMalloc((void**)&qnoffset, qnum * sizeof(int));
	CreateNbsOffsetArrayCudaS(qnum, qncount, qnoffset); //offset//
	NbsNum = SumSPixelsCuda(qnum, qncount); //get count of all candidate neighbors for all qs//
	

	// make vertex and distance arrays(of size sum of sums)
	cudaMalloc((void**)&NbVertexDist, NbsNum * sizeof(unsigned long long));
	cudaMalloc((void**)&NbVertex, NbsNum * sizeof(int));

	FillAllWithValue(qncount, qnum, 0); //zero qncount
	FillDistanceCudaS(qnum, sPixelsNum, qscount, qsoffset, qkfound, globalW, globalH, matrixVM, matrixPVM, vpos, qrads, cellWidth, xfcount, xfoffset, FragVertex, FragDepth, qncount, qnoffset, NbVertex, NbVertexDist); //fill//
	
	SortNeighborsCudaS(NbsNum, NbVertex, NbVertexDist); //sort//

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "num of all nbs for all qs: " << NbsNum << "\n";

	cout << "***kNN search time: " << milliseconds << '\n';

	Nbs.resize(qnum); for (int i = 0; i < qnum; i++) { Nbs[i].resize(k + 3); for (int j = 0; j < k + 3; j++) Nbs[i][j] = -1; }
	CopyKNeighborsCudaS(qnum, qncount, NbsNum, NbVertex, Nbs);

	


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




