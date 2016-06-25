///////////////////////////////////////////////////////////////////////
// by Wei Zhang (wzhang009 at gmail.com), June, 23, 2011

#include "mex.h"
// #include <string.h>

//////////////////////////////////////////////////////////////////////////////////////////////////////
// this function is used by gacPathAffinity_fast_c.cpp and gacInitAffinityTable_knn_c.cpp
double gdlComputeDirectedAffinity (double *pW, const int height, const mxArray *cluster_i, const mxArray *cluster_j)
{
    int num_i = mxGetNumberOfElements (cluster_i);
    int num_j = mxGetNumberOfElements (cluster_j);
    double *pCi = mxGetPr(cluster_i);
    double *pCj = mxGetPr(cluster_j);
    
    // L_ij = graphW(cluster_i, cluster_j);
    // L_ji = graphW(cluster_j, cluster_i);
    // computing sum(L_ij,1)*sum(L_ji,2)
    double sum1 = 0;
    for (int j = 0; j < num_j; j++) {
        int index_j = (int(pCj[j]-0.5));  // -1
        double *pW1 = pW+index_j*height;
        double *pW2 = pW+index_j;
        double Lij = 0;
        double Lji = 0;
        for (int i = 0; i < num_i; i++) {
            int index_i = int(pCi[i]-0.5);  // -1
            Lij += pW1[index_i]; // pW[index_i+index_j*height];
            Lji += pW2[index_i*height]; // pW[index_j+index_i*height];
        }
        sum1 += Lij*Lji;
    }

    return sum1 / (num_i*num_i);
}