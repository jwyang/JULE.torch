///////////////////////////////////////////////////////////////////////
// by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

#include "mex.h"
// #include <string.h>

//#define SAFEMXDESTROYARRAY(p) { if (p != NULL) { mxDestroyArray(p); p = NULL; } }
//#define SAFEDELETEARRAY(p) { if (p != NULL) { delete []p; p = NULL; } }
#define MYINF 1e10

//////////////////////////////////////////////////////////////////////////////////////////////////////
// function L = gacPartialMin_knn_c (affinityTab, curGroupNum, KcCluster)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
    const mxArray *affinityTab = prhs[0];
    int curGroupNum = int(*mxGetPr(prhs[1]) + 0.5);
    const mxArray *KcCluster = NULL;
    if (nrhs > 2) {
        KcCluster = prhs[2];
    }

    int numClusters = mxGetM(affinityTab);  // height of a matlab matrix
    if (mxGetNumberOfDimensions(affinityTab) != 2 || numClusters != mxGetN(affinityTab) || numClusters < curGroupNum) {
		mexErrMsgTxt("affinityTab is not valid!");
	}
    double *pAffinityTab = mxGetPr(affinityTab);
    
    int Kc = 0;
    double *pKcCluster = NULL;
    if (KcCluster != NULL) {
        Kc = mxGetM(KcCluster);
        if (mxGetNumberOfDimensions(KcCluster) != 2 || curGroupNum > mxGetN(KcCluster)) {
            mexErrMsgTxt("KcCluster is not valid!");
        }
        pKcCluster = mxGetPr(KcCluster);
    }
    
    int minIndex1 = 0;
    int minIndex2 = 0;
    double minElem = MYINF;
    double *pAffinityTab_j = pAffinityTab;
    
    if (pKcCluster == NULL || curGroupNum < 1.2*Kc) {
        for (int j = 0; j < curGroupNum; j++) {
            for (int i = 0; i < curGroupNum; i++) {
                if (pAffinityTab_j[i] < minElem) {
                    minElem = pAffinityTab_j[i];
                    minIndex1 = i;
                    minIndex2 = j;
                }
            }
            pAffinityTab_j += numClusters;
        }
    }
    else {
        double *pKcCluster_j = pKcCluster;
        for (int j = 0; j < curGroupNum; j++) {
            for (int i = 0; i < Kc; i++) {
                const int index_i = int(pKcCluster_j[i]-0.5);  // -1
                if (index_i >= 0 && index_i < curGroupNum) {
                    if (pAffinityTab_j[index_i] < minElem) {
                        minElem = pAffinityTab_j[index_i];
                        minIndex1 = index_i;
                        minIndex2 = j;
                    }
                }
            }
            pAffinityTab_j += numClusters;
            pKcCluster_j += Kc;
        }
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