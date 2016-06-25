///////////////////////////////////////////////////////////////////////
// by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

#include "mex.h"
#include <vector>
#include <stack>
using namespace std;

#define SAFEDELETEARRAY(pointer) { if(pointer) { delete []pointer; pointer = NULL; } }

void CLLink (int N, const double *NNIdx, vector<vector<int> > &outputClusters);

//////////////////////////////////////////////////////////////////////////////////////////////////////
// outputClusters = gacOnelink_c (NNIndex)
//      a fast version of L-links for p = 1
void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) 
{
	vector<vector<int> > outputClusters;
    
    if (nrhs != 1 || nlhs != 1) {
        mexErrMsgTxt("Example use: outputClusters = gacOnelink_c (NNIndex)");
    }
    
	const int vin = 0;
	double *fpt = mxGetPr(prhs[vin]);
	const int Dim = mxGetNumberOfDimensions(prhs[vin]);
	if ((Dim == 2 && mxGetN(prhs[vin]) < 2) || (Dim != 2)) {
        mexErrMsgTxt("NNIndex is not valid: should be a N x K (K >= 2) matrix!");
	}
    int N = mxGetM(prhs[vin]);
    const double *NNIdx = mxGetPr(prhs[vin]) + N;

	CLLink (N, NNIdx, outputClusters);

	///* Create the array */
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
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Entry
void CLLink (int N, const double *NNIdx, vector<vector<int> > &outputClusters)
{
	int *visited = new int [N];
    for (int i = 0; i < N; i++)  { visited[i] = -1; }

    int count = 0;
    for (int i = 0; i < N; i++)
	{
        stack<int> linkedIdx;
        int cur_idx = i;
        while (visited[cur_idx] == -1) {
            linkedIdx.push(cur_idx);
			visited[cur_idx] = -2;  // -2 is for visited but not assigned
            cur_idx = int(NNIdx[cur_idx] - 0.5);
        }
        if (visited[cur_idx] < 0)  { visited[cur_idx] = count;  count++; }
		int cluster_id = visited[cur_idx];
        while (!linkedIdx.empty()) {
            visited[linkedIdx.top()] = cluster_id;
            linkedIdx.pop();
        }
	}

    outputClusters.clear();
	vector<int> tmpVec;
    for (int i = 0; i < count; i++)
	{
		tmpVec.clear();
        outputClusters.push_back(tmpVec);
    }
	for (int i = 0; i < N; i++)
	{
		outputClusters[visited[i]].push_back(i);  // -1
    }

    SAFEDELETEARRAY(visited);
}
