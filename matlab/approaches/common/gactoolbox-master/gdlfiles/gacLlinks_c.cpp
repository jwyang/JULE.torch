///////////////////////////////////////////////////////////////////////
// by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

#include "mex.h"
#include <vector>
#include <iterator>
#include <algorithm>
using namespace std;

#define SAFEDELETEARRAY(pointer) { if(pointer) { delete []pointer; pointer = NULL; } }

inline void swap(int *a, int *b) { int tmp = *a; *a = *b; *b = tmp; }
void mergeLLinks (int *LLIdx, int numLLIdx, vector<vector<int> > &outputClusters);
void CLinkageRanking (const double *dist_mat, int N, int *NNIdx, int numK, int *labeledIdx, int numLabeled, int numToBeRanked, int *LLIdx);
void CLLink (const double *dist_mat, int N, int *NNIdx, int K, int p, vector<vector<int> > &outputClusters);

//////////////////////////////////////////////////////////////////////////////////////////////////////
// outputClusters = gacLlinks_c (distance_matrix, NNIndex, p)
void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) 
{
	int *NNIdx = NULL;
	vector<vector<int> > outputClusters;
	int N, K, p, Dim, vin;

	vin = 0;
	Dim = mxGetNumberOfDimensions(prhs[vin]);
	if (Dim == 2) {
		N = mxGetM(prhs[vin]);
		if (N != mxGetN(prhs[vin])) {
			mexErrMsgTxt("distance_matrix is not square!");
		}
	}
	else {
		mexErrMsgTxt("distance_matrix is not a matrix!");
	}
    const double *dist_mat = mxGetPr(prhs[vin]);

	vin = 1;
	double *fpt = mxGetPr(prhs[vin]);
	Dim = mxGetNumberOfDimensions(prhs[vin]);
	if (Dim == 2) {
		K = mxGetN(prhs[vin]);
		if (N == mxGetM(prhs[vin])) {
			NNIdx = new int [N*K];
			for (int j = 0; j < N*K; j++) { NNIdx[j] = int(fpt[j] - 0.5); } // - 1
		}
		else {
			mexErrMsgTxt("NNIndex does not match distance_matrix in size!");
		}
	}
	else {
		mexErrMsgTxt("NNIndex is not a matrix!");
	}

	vin = 2;
	p = int (mxGetScalar(prhs[vin]) + 0.5);

	CLLink (dist_mat, N, NNIdx, K, p, outputClusters);

	///* Create the array */
	//plhs[0] = mxCreateDoubleMatrix(1,3,mxREAL);
	const int outDim[1] = {outputClusters.size()};
	plhs[0] = mxCreateCellArray (1, outDim);
	for (int idx = 0; idx < outDim[0]; idx++) {
		mxArray *tVal = mxCreateDoubleMatrix(outputClusters[idx].size(),1,mxREAL);
		double *tValEntry = mxGetPr(tVal);
		for (int j = 0; j < outputClusters[idx].size(); j++) {
			tValEntry[j] = outputClusters[idx][j] + 1; // plus 1
		}
		mxSetCell (plhs[0], idx, tVal);
	}

	SAFEDELETEARRAY(NNIdx);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Entry
// K: including the point itself
void CLLink (const double *dist_mat, int N, int *NNIdx, int K, int p, vector<vector<int> > &outputClusters)
{
	int *localNNIdx = new int [K];
	int *localLLIdx = new int [K];
	int *locallabeledIdx = new int [1];
	locallabeledIdx[0] = 0;

	outputClusters.clear();
	for (int i = 0; i < N; i++)
	{
		// nearest neighbor set \{i, S_i^{2K}\}
		for (int j = 0; j < K; j++)  { localNNIdx[j] = NNIdx[i+j*N]; }
		// rank the points using the NNs
		CLinkageRanking (dist_mat, N, localNNIdx, K, locallabeledIdx, 1, p, localLLIdx);
		// merge new l-link cluster to clusters
		mergeLLinks(localLLIdx, p+1, outputClusters);
	}
	SAFEDELETEARRAY(localNNIdx);
	SAFEDELETEARRAY(localLLIdx);
	SAFEDELETEARRAY(locallabeledIdx);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////
// function outputClusters = mergeLinks(iNNIndex, outputClusters)
// find and merge l-link clusters
void mergeLLinks (int *LLIdx, int numLLIdx, vector<vector<int> > &outputClusters)
{
	vector<int> vecLLIdx(numLLIdx);
	for (int i = 0; i < numLLIdx; i++)  { vecLLIdx[i] = LLIdx[i]; }
	sort(vecLLIdx.begin(), vecLLIdx.end());
	int first_merge = -1; // record the index of first merging
	vector<int> tmpVec;

	for (int j = 0; j < outputClusters.size(); j++) {
		tmpVec.clear();
		set_intersection(outputClusters[j].begin(), outputClusters[j].end(), vecLLIdx.begin(), vecLLIdx.end(), back_inserter(tmpVec));
		if (tmpVec.size() > 0)  // intersection is not empty
		{
			tmpVec.clear();
			if (first_merge < 0) {
				set_union(outputClusters[j].begin(), outputClusters[j].end(), vecLLIdx.begin(), vecLLIdx.end(), back_inserter(tmpVec));
				outputClusters[j] = tmpVec;
				first_merge = j;
			}
			else {
				set_union(outputClusters[j].begin(), outputClusters[j].end(), outputClusters[first_merge].begin(), outputClusters[first_merge].end(), back_inserter(tmpVec));
				outputClusters[first_merge] = tmpVec;
				outputClusters.erase(outputClusters.begin()+j);  // remove Cluster j as it is merged to another cluster
				j--;
			}
		}
	}

	if (first_merge < 0) {
		outputClusters.push_back(vecLLIdx);
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
// function [sortedIndex] = LinkageRanking(distance_matrix, labeledIndex, number_rankedPoint)
// ranking by Linkage - find the nearest point and then pass message using NN 
void CLinkageRanking (const double *dist_mat, int N, int *NNIdx, int numK, int *labeledIdx, int numLabeled, int numToBeRanked, int *LLIdx)
{
	int numOutput = numLabeled+numToBeRanked;
	for (int k = 0; k < numK; k++) { LLIdx[k] = NNIdx[k]; }
	for (int k = 0; k < numLabeled; k++) { swap(LLIdx+k, LLIdx+labeledIdx[k]); }
	if (numToBeRanked > 0) {
		for (int k = 0; k < numToBeRanked; k++) {
			// nearest neighbor search
			double minDist = 1e10;
			int minIdx = k;
			for (int j = numLabeled+k; j < numK; j++) {
				for (int i = 0; i < numLabeled+k; i++) {
					if (dist_mat[LLIdx[i]+LLIdx[j]*N] < minDist) {
						minDist = dist_mat[LLIdx[i]+LLIdx[j]*N];
						minIdx = j;
					}
				}
			}
			swap(LLIdx+k, LLIdx+minIdx);
		}
	}
}