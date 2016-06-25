/**********
 * 
 * 
 *  mex interface to set to zero the main diagonal of a matrix A
 * 
 *  To mexify:
 *             mex  zero_diag.cpp;
 *
 *
 *  [A0] = zero_diag(A);
 *
 *  Input: 
 *    A = input matrix
 *
 *  Output:
 *    A0 = same as A with main diagonal set to 0
 *
 *
 *  Lihi Zelnik (Caltech) May.2006
 * 
 * 
 ************/

#include "mex.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/* ////////////// macros to address input  */
#define INMATRIX  prhs[0]

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
    if (nrhs < 1) {
        mexErrMsgTxt("Too few input arguments.");
        return;
    }
    if (nrhs > 1) {
        mexErrMsgTxt("Too many input arguments.");
        return;
    }
    
    if( DEBUG )
        mexPrintf("Just starting\n");
        
     
    /* check matrix dimensions */
    const int *idims =  mxGetDimensions(INMATRIX);
    if(idims[0] != idims[1])
        mexErrMsgTxt("Input matrix must be square");
    const int node_num = idims[0];
    int i,j;
    
    if(mxIsSparse(INMATRIX)) {   /* Sparse matrix handling */
                      
        if( DEBUG )
            mexPrintf("Detected the matrix is sparse with %d nodes\n",node_num);
        
        /* Pointers to non zero rows, cols and values */
        int *in_rows = mxGetIr(INMATRIX);
        int *in_cols = mxGetJc(INMATRIX);
        double *in_values = mxGetPr(INMATRIX);
        int non_zero = mxGetNzmax(INMATRIX);
        
        if( DEBUG )
            mexPrintf("There are %d non zeros\n",non_zero);
       
        /* prepare output array */
        double *out_values;        
        plhs[0] = mxDuplicateArray(prhs[0]);
        out_values = mxGetPr(plhs[0]);        

        int starting_row_index, stopping_row_index, current_index;
        /* Go over the input matrix and set its main diagonal to 0 */
        for (j=0; j<node_num; j++)  {   /* loop over all columns */
            starting_row_index = in_cols[j];
            stopping_row_index = in_cols[j+1];
            if (starting_row_index == stopping_row_index) { /* empty column */
                continue;
            }
            else {  /* loop over all rows for current column */
                for (current_index = starting_row_index;
                current_index < stopping_row_index;
                current_index++)  {                   
                    if( j == in_rows[current_index] )
                        out_values[current_index] = 0;
                    else
                        out_values[current_index] = in_values[current_index];
                    
                }
            }
        }
        if( DEBUG )
            mexPrintf("All matrix processed safely, current_index=%d\n",current_index);
        
    }
    /*******************   FULL MATRICES ******************************/
    else { /* Full matrix handling  */
        
        if( DEBUG )
            mexPrintf("Detected the matrix is full with %d nodes\n",node_num);
        
        /* Pointers to values */
        double *in_values = mxGetPr(INMATRIX);
                                     
        /* prepare output arrays */
        double *out_values;        
        plhs[0] = mxDuplicateArray(prhs[0]);
        out_values = mxGetPr(plhs[0]);
                
        /* prepare value arrays */
        /* Go over the input matrix and set its main diagonal to 0 */
        int total = 0;
        for (j=0; j<node_num; j++)  {   /* loop over all columns */
            for (i=0; i<node_num; i++ ){            
                if( i == j )
                    out_values[total] = 0;
                else
                    out_values[total] = in_values[total];
                total++;
            }
        }       
        if( DEBUG )
            mexPrintf("All matrix processed safely, total=%d\n",total);           
    }
    
    return;
}


