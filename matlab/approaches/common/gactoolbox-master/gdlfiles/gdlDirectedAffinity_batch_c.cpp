///////////////////////////////////////////////////////////////////////
// by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

#include "mex.h"
// #include <string.h>

//#define SAFEMXDESTROYARRAY(p) { if (p != NULL) { mxDestroyArray(p); p = NULL; } }
//#define SAFEDELETEARRAY(p) { if (p != NULL) { delete []p; p = NULL; } }
double gdlComputeDirectedAffinity (double *pW, const int height, const mxArray *cluster_i, const mxArray *cluster_j);

//////////////////////////////////////////////////////////////////////////////////////////////////////
// function [L_i] =  = gdlDirectedAffinity_batch_c (graphW, initClusters, i)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
    if (nrhs != 3) {
        mexErrMsgTxt("Wrong number of input!");
    }

    const mxArray *graphW = prhs[0];
    const mxArray *initClusters = prhs[1];
    double *pI = mxGetPr(prhs[2]);
    int i = int(pI[0] - .5);
    
    if (!mxIsCell(initClusters)) {
		mexErrMsgTxt("initClusters is not a cell!");
    }
    int numClusters = mxGetNumberOfElements (initClusters);
    
    const int height = mxGetM(graphW);  // height of a matlab matrix
    if (mxGetNumberOfDimensions(graphW) != 2 || height != mxGetN(graphW)) {
		mexErrMsgTxt("graphW is not a square matrix!");
	}
    double *pW = mxGetPr(graphW);
 
    // output:
	plhs[0] = mxCreateDoubleMatrix(1,numClusters,mxREAL);
    double *pAsymAffTab = mxGetPr(plhs[0]);
    for (int j = 0; j < numClusters; j++) {
        if (j == i) {
            pAsymAffTab[j] = - 1e10;
        }
        else {
            pAsymAffTab[j] = gdlComputeDirectedAffinity (pW, height, mxGetCell (initClusters, i), mxGetCell (initClusters, j));
        }
    }
}