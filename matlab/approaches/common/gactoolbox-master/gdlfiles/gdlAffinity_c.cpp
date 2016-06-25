///////////////////////////////////////////////////////////////////////
// by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

#include "mex.h"
// #include <string.h>

//#define SAFEMXDESTROYARRAY(p) { if (p != NULL) { mxDestroyArray(p); p = NULL; } }
//#define SAFEDELETEARRAY(p) { if (p != NULL) { delete []p; p = NULL; } }
double gdlComputeAffinity (double *pW, const int height, const mxArray *cluster_i, const mxArray *cluster_j, double *AsymAff);

//////////////////////////////////////////////////////////////////////////////////////////////////////
// function [L, L_i, L_j] =  = gdlAffinity_c (graphW, cluster_i, cluster_j)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
    const mxArray *graphW = prhs[0];
    const mxArray *cluster_i = prhs[1];
    const mxArray *cluster_j = prhs[2];

    const int height = mxGetM(graphW);  // height of a matlab matrix
    if (mxGetNumberOfDimensions(graphW) != 2 || height != mxGetN(graphW)) {
		mexErrMsgTxt("graphW is not a square matrix!");
	}
    double *pW = mxGetPr(graphW);
 
    // output:
    double AsymAff[2];
	plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    *mxGetPr(plhs[0]) = gdlComputeAffinity (pW, height, cluster_i, cluster_j, AsymAff);
	plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
	plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
    *mxGetPr(plhs[1]) = AsymAff[0];
    *mxGetPr(plhs[2]) = AsymAff[1];
}