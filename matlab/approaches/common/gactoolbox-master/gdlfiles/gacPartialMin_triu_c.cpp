///////////////////////////////////////////////////////////////////////
// by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

#include "mex.h"
// #include <string.h>

//#define SAFEMXDESTROYARRAY(p) { if (p != NULL) { mxDestroyArray(p); p = NULL; } }
//#define SAFEDELETEARRAY(p) { if (p != NULL) { delete []p; p = NULL; } }
#define MYINF 1e10

//////////////////////////////////////////////////////////////////////////////////////////////////////
// function L = gacPartialMin_triu_c (affinityTab, curGroupNum)
void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) 
{
    const mxArray *affinityTab = prhs[0];
    int curGroupNum = int(*mxGetPr(prhs[1]) + 0.5);

    int numClusters = mxGetM(affinityTab);  // height of a matlab matrix
    if (mxGetNumberOfDimensions(affinityTab) != 2 || numClusters != mxGetN(affinityTab) || numClusters < curGroupNum) {
		mexErrMsgTxt("affinityTab is not valid!");
	}
    double *pAffinityTab = mxGetPr(affinityTab);
 
    int minIndex1 = 0;
    int minIndex2 = 0;

    double minElem = MYINF;
    double *pAffinityTab_j = pAffinityTab;
    for (int j = 0; j < curGroupNum; j++) {
        for (int i = 0; i < j; i++) {
            if (pAffinityTab_j[i] < minElem) {
                minElem = pAffinityTab_j[i];
                minIndex1 = i;
                minIndex2 = j;
            }
        }
        pAffinityTab_j += numClusters;
    }
    if (minIndex1 > minIndex2) {
        int tmp = minIndex1;
        minIndex1 = minIndex2;
        minIndex2 = tmp;
    }

    // output:
	plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
	plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
	plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL);
    *mxGetPr(plhs[0]) = minElem;
    *mxGetPr(plhs[1]) = minIndex1+1;  // +1
    *mxGetPr(plhs[2]) = minIndex2+1;  // +1
}