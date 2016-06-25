///////////////////////////////////////////////////////////////////////
// by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

#include "mex.h"
// #include <string.h>

//#define SAFEMXDESTROYARRAY(p) { if (p != NULL) { mxDestroyArray(p); p = NULL; } }
//#define SAFEDELETEARRAY(p) { if (p != NULL) { delete []p; p = NULL; } }
double gdlComputeDirectedAffinity (double *pW, const int height, const mxArray *cluster_i, const mxArray *cluster_j);

//////////////////////////////////////////////////////////////////////////////////////////////////////
// function [L_i] =  = gdlDirectedAffinity_c (graphW, initClusters, i, j)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
    if (nrhs != 4) {
        mexErrMsgTxt("Wrong number of input!");
    }

    const mxArray *graphW = prhs[0];
    const mxArray *initClusters = prhs[1];
    double *pI = mxGetPr(prhs[2]);
    double *pJ = mxGetPr(prhs[3]);
    int i = int(pI[0] - .5);
    int j = int(pJ[0] - .5);

    const int height = mxGetM(graphW);  // height of a matlab matrix
    if (mxGetNumberOfDimensions(graphW) != 2 || height != mxGetN(graphW)) {
		mexErrMsgTxt("graphW is not a square matrix!");
	}
    double *pW = mxGetPr(graphW);
 
    // output:
	plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    *mxGetPr(plhs[0]) = gdlComputeDirectedAffinity (pW, height, mxGetCell (initClusters, i), mxGetCell (initClusters, j));
}