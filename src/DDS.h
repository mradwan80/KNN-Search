#pragma once

//#include <GL/glew.h>
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include <thrust/host_vector.h>
#include <vector_types.h>

#include <vector>
using namespace std;

//x: pixel
//f: fragment
//n: neighbor (also a fragment and a vertex)

#define blocksize 512
#define maxblocks 65535

void checkCUDAError(const char* msg);

struct pointCoords
{
	float x, y, z;
};

class DDS
{

private:

	bool debug_details;

	vector<pointCoords>* vxPos;

	float viewWidth;
	glm::mat4 vmMat, pvmMat;
	
	int FragsNum;

	float* matrixPVM;
	float* matrixVM;
	float* vpos;
	int* xfcount;	int* xfoffset;
	float* FragDepth; int* FragVertex; unsigned long long* FragDepthPixel; //in case no sorting, only FragVertex is important//
	float* FragX; float* FragY; float* FragZ;

	void PrepareInput();
	void CountFrags();
	void CreateOffsetAndFragsVectors(bool DoSortFrags);
	void ProjectFrags(bool DoSortFrags);
	void SortFrags();
	void GetFragsCoords();
	void FreeMemory();

	void kNNsearch(int k, float SearchRad);
	void kNNsearchWithSort(int k, float SearchRad);
	//void kNNsearchWithSortVarRad(int k, float SearchRad);
	
	//knn stuff//
	int qnum;
	int len;
	int NbsNum;
	unsigned long long* NbVertexDist; int* NbVertex;
	vector<vector<int> > Nbs;
	int* qncount;	int* qnoffset;

	//varying rad stuff
	//int* qscount; int* qsoffset;
	//float* qrads;
	//bool* qkfound;


public:

	int globalW, globalH;


	DDS();
	DDS(int w, int h, float viewWidthI, vector<pointCoords>* Pos, glm::mat4 vmMatI, glm::mat4 pvmMatI);
	~DDS();

	float BuildDDS(bool DoSortFrags = true, bool debug_details = false);
	void findKNN(int k, float SearchRad, bool DoSortFrags);

};

void FillAllWithValue(int* arr, int sz, int val);
void FillAllWithValue(float* arr, int sz, float val);
void FillAllWithValue(unsigned long long* arr, int sz, unsigned long long val);
void CountFragsCuda(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, int* xfcount);
void SetOffsetVectorCuda(int pxNum, int* xfcount, int* xfoffset);
int GetFragsNumCuda(int vxNum, int* xfcount);
void ProjectFragsCuda(int vxNum, int globalW, int globalH, float viewWidth, float* vmMat, float* pvmMat, float* vpos, int* xfcount, int* xfoffset, float* FragDepth, int* FragVertex, unsigned long long* FragDepthPixel, bool DoSortFrags);
void SortFragsCuda(int FragsNum, float* FragDepth, int* FragVertex, unsigned long long* FragDepthPixel);

//use a template?
void FillAllWithValue(int* arr, int sz, int val);
void FillAllWithValue(float* arr, int sz, float val);
void FillAllWithValue(bool* arr, int sz, bool val);
void FillAllWithValue(unsigned long long* arr, int sz, unsigned long long val);

//knn functions//
void GetFragsCoordsCuda(int N, int* FragVertex, float* vpos, float* FragX, float* FragY, float* FragZ);
void CopyCountsCuda(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragX, float* FragY, float* FragZ, int* qncount);
void CreateNbsOffsetArrayCuda(int n, int* qncount, int* qnoffset);
int SumNbsCuda(int n, int* qncount);
void FillDistanceCuda(int qnum, int len, float searchRad, int globalW, int globalH, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragX, float* FragY, float* FragZ, int* FragVertex, int* qnoffset, int* NbVertex, unsigned long long* NbVertexDist);
void SortNeighborsCuda(int NbsNum, int* NbVertex, unsigned long long* NbVertexDist);
void CopyKNeighborsCuda(int k, float SearchRad, int qnum, int len, int* qncount, int NbsNum, int* NbVertex, int vnum, float* vpos, vector<vector<int> >& Nbs);

//knn with sort functions//
void CopyCountsCudaS(int qnum, int len, float searchRad, float cellWidth, int globalW, int globalH, float* vmMat, float* pvmMat, int* xfcount, int* xfoffset, float* vpos, float* FragDepth, int* qncount);
void FillDistanceCudaS(int qnum, int len, float searchRad, float cellWidth, int globalW, int globalH, float* vmMat, float* pvmMat,int* xfcount, int* xfoffset, float* vpos, float* FragX, float* FragY, float* FragZ, int* FragVertex, float* FragDepth, int* qnoffset, int* NbVertex, unsigned long long* NbVertexDist);

//knn with sort and varying rad functions//
//SV: sort and var//
/*void CalculateSquareSizeCuda(int qnum, float* qrads, bool* qkfound, float cellWidth, int* qscount);
void CreateSquaresOffsetArrayCuda(int n, int* qscount, int* qsoffset);
int SumSPixelsCuda(int n, int* qscount);
void CountNeighborsCuda(int qnum, int sPixelsNum, int* qscount, int* qsoffset, bool* qkfound, int globalW, int globalH, float* vmMat, float* pvmMat, float* vpos, float* qrads, float cellWidth, int* xfcount, int* xfoffset, float* FragDepth, int* qncount);
void CreateNbsOffsetArrayCudaSV(int n, int* qncount, int* qnoffset);
int SumNbsCudaSV(int n, int* qncount);
void FillDistanceCudaSV(int qnum, int sPixelsNum, int* qscount, int* qsoffset, bool* qkfound, int globalW, int globalH, float* vmMat, float* pvmMat, float* vpos, float* qrads, float cellWidth, int* xfcount, int* xfoffset, int* FragVertex, float* FragDepth, int* qncount, int* qnoffset, int* NbVertex, unsigned long long* NbVertexDist);
void UpdateRadsCuda(int qnum, int k, int* qncount, bool* qkfound, float* qrads);
bool AllKNbsFoundCuda(int qnum, bool* qkfound);
void SortNeighborsCudaSV(int NbsNum, int* NbVertex, unsigned long long* NbVertexDist);
void CopyKNeighborsCudaSV(int qnum, int* qncount, int NbsNum, int* NbVertex, vector<vector<int>>& Nbs);*/


