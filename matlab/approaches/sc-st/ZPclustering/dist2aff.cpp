/**********
 * 
 * 
 *  mex interface to compute the affinity matrix of a given distance matrix
 *  (for some reason this is slow in matlab)
 * 
 * To mexify:
 *              mex  dist2aff.cpp;
 *
 *
 *
 * A = dist2aff(D,SS);
 *
 * Input:
 *   D = distance matrix
 *   SS = scale
 *  
 * Output:
 *   A = affinity matrix
 *
 *
 *
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

/*
 *  To mexify:
 *
 mex  dist2aff.c;
 *
 *
 */

/* ////////////// macros to address input  */
#define DISTMATRIX  prhs[0]
#define SIGMA prhs[1]

#define DEBUG 0    /* set to 1 and mex to see print outs */



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
    if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
        return;
    }
  /* Make sure input number is sufficient */
    if (nrhs < 2) {
        mexErrMsgTxt("Too few input arguments.");
        return;
    }
    if (nrhs > 2) {
        mexErrMsgTxt("Too many input arguments.");
        return;
    }
    
    if( DEBUG )
        mexPrintf("Just starting\n");
    
    
    /* get the scale */
    const double S2 = (double) mxGetScalar(SIGMA)*mxGetScalar(SIGMA);
    
    if( DEBUG )
        mexPrintf("Got sigma^2 = %g\n",S2);

    
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

        int i,j;        
        int starting_row_index, stopping_row_index, current_row_index;
        int total = 0;
        int row_num,col_total;
        
        
        /* scale the distance and copy to output */
        /* prepare output arrays */        
        plhs[0] = mxDuplicateArray(prhs[0]);
        double *affinity = mxGetPr(plhs[0]);        

        /* Go over the distance matrix and scale it */
        total = 0;
        double s;
        for (j=0; j<node_num; j++)  {   /* loop over all columns */
            starting_row_index = distmat_cols[j];
            stopping_row_index = distmat_cols[j+1];
            if (starting_row_index == stopping_row_index) {
                continue;
            }
            else {  /* loop over all rows for current column */
                for (current_row_index = starting_row_index;
                current_row_index < stopping_row_index;
                current_row_index++)  {
                    affinity[total] = exp(-distmat_values[total]/S2);
                    total++;
                }
            }
        }
        if( DEBUG )
            mexPrintf("Affinities conputed for %d matrix entries\n",total);        
    }
    /*******************   FULL MATRICES ******************************/
    else { /* Full matrix handling  */
        
        if( DEBUG )
            mexPrintf("Detected the matrix is full with %d nodes\n",node_num);
        
        /* Pointers to values */
        double *distmat_values = mxGetPr(DISTMATRIX);
                
        /************* Scale the distance and copy to output */
        
        /* prepare output arrays */   
        plhs[0] = mxDuplicateArray(prhs[0]);
        double *affinity = mxGetPr(plhs[0]);
                       
        /* prepare value arrays */
        /* Go over the distance matrix and scale it */
        int i,j,total = 0;
        double s;
        for (j=0; j<node_num; j++)  {   /* loop over all columns */
            for (i=0; i<node_num; i++ ){
                affinity[total] = exp(-distmat_values[total]/S2);
                total++;
            }
        }       
        if( DEBUG )
            mexPrintf("Affinities conputed for %d matrix entries\n",total);                  
    }
    
    return;
}


