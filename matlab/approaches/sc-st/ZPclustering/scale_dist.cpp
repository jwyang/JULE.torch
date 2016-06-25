/**********
 * 
 * 
 *  mex interface to compute locally scaled distance matrix
 * 
 *  To mexify:
 *             mex  scale_dist.cpp;
 *
 *
 *  [D_LS,A_LS,LS] = scale_dist(D,n);
 *
 *  Input: 
 *    D = distance matrix
 *    n = the distance to the n'th neighbor is taken as the local scale
 *
 *  Output:
 *    D_LS = scaled distance matrix
 *    A_LS = scaled affinity matrix
 *    LS = the local scales
 *
 *
 *  Lihi Zelnik (Caltech) Feb.2005
 * 
 * 
 ************/

#include "mex.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/* ////////////// macros to address input  */
#define DISTMATRIX  prhs[0]
#define NEIGHBORIND prhs[1]

#define DEBUG 0    /* set to 1 and mex to see print outs */


/* ////////////// sort array */
typedef double T;
void mysort(T* data, int N)
{
    int     i, j;
    T       v, t;
    
    if(N<=1) return;
    
  /* Partition elements  */
    v = data[0];
    i = 0;
    j = N;
    for(;;)
    {
        while(data[++i] < v && i < N) { }
        while(data[--j] > v) { }
        if(i >= j) break;
        t = data[i]; data[i] = data[j]; data[j] = t;
    }
    t = data[i-1]; data[i-1] = data[0]; data[0] = t;
    mysort(data, i-1);
    mysort(data+i, N-i);
}


/*   //////////////// main   */

void
mexFunction (
int nlhs, mxArray* plhs[],
int nrhs, const mxArray* prhs[])
{
  /* Make sure at most two output arguments are expected */
    if (nlhs < 1) {
        mexErrMsgTxt("Too few output arguments.");
        return;
    }
    if (nlhs > 3) {
        mexErrMsgTxt("Too many output arguments.");
        return;
    }
  /* Make sure input number is sufficient */
    if (nrhs < 1) {
        mexErrMsgTxt("Too few input arguments.");
        return;
    }
    if (nrhs > 2) {
        mexErrMsgTxt("Too many input arguments.");
        return;
    }
    
    if( DEBUG )
        mexPrintf("Just starting\n");
    
    
  /* get the neighbor index */
    const int K = (int) mxGetScalar(NEIGHBORIND);
    
    if( DEBUG )
        mexPrintf("Got the neighbor index %d\n",K);

    
        /* check matrix dimensions */
    const int *idims =  mxGetDimensions(DISTMATRIX);
    if(idims[0] != idims[1])
        mexErrMsgTxt("Distance matrix must be square");
    const int node_num = idims[0];
    
    if(mxIsSparse(DISTMATRIX)) {   /* Sparse matrix handling */
                      
        if( DEBUG )
            mexPrintf("Detected the matrix is sparse with %d nodes\n",node_num);
        
    /* Pointers to non zero rows, cols and values */
        int *distmat_rows = mxGetIr(DISTMATRIX);
        int *distmat_cols = mxGetJc(DISTMATRIX);
        double *distmat_values = mxGetPr(DISTMATRIX);
        int non_zero = mxGetNzmax(DISTMATRIX);
        
        if( DEBUG )
            mexPrintf("There are %d non zeros\n",non_zero);
        
    /***************************
     * find for each node (column) the distance to the K'th neighbor
     * where K = neighbor_index
     */
        
        int i,j;
        double *values,*neighbor_dist;
        neighbor_dist = (double*)mxCalloc(node_num,sizeof(double));
        
        int starting_row_index, stopping_row_index, current_index;
        int row_num,col_total, is_main_d_set;
        for (j=0; j<node_num; j++)  {   /* loop over all columns */
            /* For each column fill in all the nonzero in the values array */
            starting_row_index = distmat_cols[j];
            stopping_row_index = distmat_cols[j+1];
            if (starting_row_index == stopping_row_index) {
                /* no elements in this column */
                neighbor_dist[j] = 0;
                continue;
            }
            else {
                row_num = stopping_row_index - starting_row_index;
                values = (double*)mxCalloc(row_num,sizeof(double));
                col_total = 0;
                for (current_index = starting_row_index;
                current_index < stopping_row_index;
                current_index++)  {
                    i = distmat_rows[current_index];                    
                    values[col_total++] = distmat_values[current_index];
                }
                /* Now sort the values array */
                mysort(values, col_total);
                
                /* Save the distance to K'th neighbor */
                if( col_total < K ){
                    neighbor_dist[j] = values[col_total-1];
                }
                else{
                    neighbor_dist[j] = values[K-1];
                }
                
                mxFree(values);
            }
        }
        if( DEBUG )
            mexPrintf("Processesed all columns safely, current_index=%d\n",current_index);
        
        /************* Now, scale the distance and copy to output */
        /* prepare output arrays */
        double *scales,*scaled_dist,*affinity;
        
        plhs[0] = mxDuplicateArray(prhs[0]);
        scaled_dist = mxGetPr(plhs[0]);
        
        if (nlhs > 1) {
            plhs[1] = mxDuplicateArray(prhs[0]);
            affinity = mxGetPr(plhs[1]);
        }
        if (nlhs > 2) {
            plhs[2] = mxDuplicateArray(prhs[0]);
            scales = mxGetPr(plhs[2]);
        }

        /* Go over the distance matrix and scale it */
        double s;
        for (j=0; j<node_num; j++)  {   /* loop over all columns */
            starting_row_index = distmat_cols[j];
            stopping_row_index = distmat_cols[j+1];
            if (starting_row_index == stopping_row_index) {
                continue;
            }
            else {  /* loop over all rows for current column */
                for (current_index = starting_row_index;
                current_index < stopping_row_index;
                current_index++)  {
                    i = distmat_rows[current_index];
                    s = sqrt(neighbor_dist[i] * neighbor_dist[j]);
                    if( s < 0.004 )
                        s = 0.004;
                    scaled_dist[current_index] = distmat_values[current_index] / s;
                    if( nlhs > 1 )
                        affinity[current_index] = exp(-scaled_dist[current_index]);
                    if( nlhs > 2 )
                        scales[current_index] = s;
                }
            }
        }
        mxFree(neighbor_dist);
        if( DEBUG )
            mexPrintf("All columns scaled safely, current_index=%d\n",current_index);
        
    }
    /*******************   FULL MATRICES ******************************/
    else { /* Full matrix handling  */
        
        if( DEBUG )
            mexPrintf("Detected the matrix is full with %d nodes\n",node_num);
        
        /* Pointers to values */
        double *distmat_values = mxGetPr(DISTMATRIX);
        
        /***************************
         * find for each node (column) the distance to the K'th neighbor
         * where K = neighbor_index
         */        
        int i,j;
        double *values,*neighbor_dist;
        neighbor_dist = (double*)mxCalloc(node_num,sizeof(double));
        values = (double*)mxCalloc(node_num,sizeof(double));
        
        int total = 0;
        for (j=0; j<node_num; j++)  {   /* loop over all columns */
            for (i=0; i<node_num; i++ ){
                values[i] = distmat_values[total++];
            }
            /* Now sort the values array */
            mysort(values, node_num);
                
            /* Save the distance to K'th neighbor */
            /* This is K and not K-1 since the distance to itself is also set */
            neighbor_dist[j] = values[K];
        }
        mxFree(values);
        if( DEBUG )
            mexPrintf("Processesed all columns safely, total=%d\n",total);
        
               
        /************* Now, scale the distance and copy to output */
        
        /* prepare output arrays */
        double *scales,*scaled_dist,*affinity;
        
        plhs[0] = mxDuplicateArray(prhs[0]);
        scaled_dist = mxGetPr(plhs[0]);
        
        if (nlhs > 1) {
            plhs[1] = mxDuplicateArray(prhs[0]);
            affinity = mxGetPr(plhs[1]);
        }
        if (nlhs > 2) {
            plhs[2] = mxDuplicateArray(prhs[0]);
            scales = mxGetPr(plhs[2]);
        }
                
        /* prepare value arrays */
        /* Go over the distance matrix and scale it */
        total = 0;
        double s;
        for (j=0; j<node_num; j++)  {   /* loop over all columns */
            for (i=0; i<node_num; i++ ){
                s = sqrt(neighbor_dist[i] * neighbor_dist[j]);
                if( s < 0.004 )
                    s = 0.004;
                scaled_dist[total] = distmat_values[total] / s;
                if( nlhs > 1 )
                    affinity[total] = exp(-scaled_dist[total]);
                if( nlhs > 2 )
                    scales[total] = s;
                total++;
            }
        }       
        mxFree(neighbor_dist);
        if( DEBUG )
            mexPrintf("All columns scaled safely, total=%d\n",total);           
    }
    
    return;
}


