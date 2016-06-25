///////////////////////////////////////////////////////////////////////
// by Wei Zhang (wzhang009 at gmail.com), Nov., 7, 2011

#include "mex.h"
// #include <string.h>

//#define SAFEMXDESTROYARRAY(p) { if (p != NULL) { mxDestroyArray(p); p = NULL; } }
#define MYINF 1e10
double computeAverageDegreeAffinity (double *pW, const int height, mxArray *cluster_i, mxArray *cluster_j);
double gdlComputeAffinity (double *pW, const int height, const mxArray *cluster_i, const mxArray *cluster_j, double *AsymAff);

//////////////////////////////////////////////////////////////////////////////////////////////////////
// function [affinityTab, AsymAffTab] = gdlInitAffinityTable_knn_c (graphW, initClusters, Kc)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
    const mxArray *graphW = prhs[0];
    const mxArray *initClusters = prhs[1];
    const mxArray *Kc = prhs[2];

    if (nrhs != 3) {
        mexErrMsgTxt("Wrong number of input!");
    }
    if (mxGetNumberOfDimensions(graphW) != 2 || mxGetM(graphW) != mxGetN(graphW)) {
		mexErrMsgTxt("graphW is not a square matrix!");
	}
    double *pW = mxGetPr(graphW);
    const int height = mxGetM(graphW);
 
    if (!mxIsCell(initClusters)) {
		mexErrMsgTxt("initClusters is not a cell!");
    }
    int numClusters = mxGetNumberOfElements (initClusters);
    
    // output:
	plhs[0] = mxCreateDoubleMatrix(numClusters,numClusters,mxREAL);
    mxArray *affinityTab = plhs[0]; // reference
    double *affinityTabEntry = mxGetPr(affinityTab);
    for (int i = 0; i < numClusters*numClusters; i++) { affinityTabEntry[i] = - MYINF; }
	plhs[1] = mxCreateDoubleMatrix(numClusters,numClusters,mxREAL);
    mxArray *AsymAffTab = plhs[1]; // reference
    double *AsymAffTabEntry = mxGetPr(AsymAffTab);
    for (int i = 0; i < numClusters*numClusters; i++) { AsymAffTabEntry[i] = - MYINF; }
    
    // computing approximation
    double *pTable_j = affinityTabEntry;
    for (int j = 0; j < numClusters; j++) {
        mxArray *cluster_j = mxGetCell (initClusters, j);  // cluster j
        for (int i = 0; i < j; i++) {  // note: pass negative affinity
            pTable_j[i] = - computeAverageDegreeAffinity (pW, height, mxGetCell (initClusters, i), cluster_j);  // affinityTabEntry[i+j*numClusters]
        }
        pTable_j[j] = - MYINF;
        pTable_j += numClusters;
    }
    // from upper triangular to full symmetric
    // affinityTab = triu(affinityTab) + triu(affinityTab)';
    pTable_j = affinityTabEntry;
    for (int j = 0; j < numClusters; j++) {
        for (int i = j+1; i < numClusters; i++) {  // note: pass negative affinity
            pTable_j[i] = affinityTabEntry[j+i*numClusters];  // affinityTabEntry[i+j*numClusters]
        }
        pTable_j += numClusters;
    }
    // sort
    mxArray *subplhs[1] = {NULL};
    mxArray *KcNew = mxDuplicateArray(Kc);
    mxArray *subprhs[2] = {affinityTab, KcNew};
    mxArray *inKcCluster = NULL;
    bool *pInKcCluster = NULL;
    if (mexCallMATLAB(1, subplhs, 2, subprhs, "gacFindKcCluster") == 0) {
        inKcCluster = subplhs[0];
        pInKcCluster = mxGetLogicals(inKcCluster);
    }

    // computing
    double tmpAsymAff[2];
    pTable_j = affinityTabEntry;
    bool *pInKcCluster_j = pInKcCluster;
    for (int j = 0; j < numClusters; j++) {
        mxArray *cluster_j = mxGetCell (initClusters, j);  // cluster j
        for (int i = 0; i < j; i++) {
            if (pInKcCluster_j[i]) {
                pTable_j[i] = gdlComputeAffinity(pW, height, mxGetCell (initClusters, i), cluster_j, tmpAsymAff);  // affinityTabEntry[i+j*numClusters]
                AsymAffTabEntry[i+j*numClusters] = tmpAsymAff[0];
                AsymAffTabEntry[j+i*numClusters] = tmpAsymAff[1];
            }
            else {
                pTable_j[i] = - MYINF;  // affinityTabEntry[i+j*numClusters]
            }
        }
        pTable_j += numClusters;
        pInKcCluster_j += numClusters;
    }
    // from upper triangular to full symmetric
    // affinityTab = triu(affinityTab) + triu(affinityTab)';
    pTable_j = affinityTabEntry;
    for (int j = 0; j < numClusters; j++) {
        for (int i = j+1; i < numClusters; i++) {  // note: pass negative affinity
            pTable_j[i] = affinityTabEntry[j+i*numClusters];  // affinityTabEntry[i+j*numClusters]
        }
        pTable_j += numClusters;
    }
    
    mxDestroyArray(inKcCluster);
}

double computeAverageDegreeAffinity (double *pW, const int height, mxArray *cluster_i, mxArray *cluster_j)
{
    int num_i = mxGetNumberOfElements (cluster_i);
    int num_j = mxGetNumberOfElements (cluster_j);
    double *pCi = mxGetPr(cluster_i);
    double *pCj = mxGetPr(cluster_j);
    // L_ij = graphW(cluster_i, cluster_j);
    // L_ji = graphW(cluster_j, cluster_i);
    // computing sum(sum(L_ij)), sum(sum(L_ji))
    double sumLij = 0;
    double sumLji = 0;
    for (int j = 0; j < num_j; j++) {
        int index_j = (int(pCj[j]-0.5));  // -1
        double *pW1 = pW+index_j*height;
        double *pW2 = pW+index_j;
        for (int i = 0; i < num_i; i++) {
            int index_i = int(pCi[i]-0.5);  // -1
            sumLij += pW1[index_i]; // pW[index_i+index_j*height];
            sumLji += pW2[index_i*height]; // pW[index_j+index_i*height];
        }
    }

    return (sumLij + sumLji) / (num_i*num_j);
}
