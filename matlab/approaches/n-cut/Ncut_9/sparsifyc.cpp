/*=================================================================
* syntax: SPMX = SPARSIFY(MX, THRES)
*
* SPARSIFY  - sparsify the input matrix, i.e. ignore the values 
* 			of the matrix which	are below a threshold
* 			
* 			Input: 	- MX: m-by-n matrix (sparse or full)
* 					- THRES: threshold value (double)
*
* 			Output: - SPMX: m-by-n sparse matrix only with values 
* 					whose absolut value is above the given threshold
*
* 			Written by Mirko Visontai (10/24/2003)
*=================================================================*/


#include <math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Declare variable */
    int i,m,n,nzmax,newnnz,col,processed,passed;
	int starting_row_index, current_row_index, stopping_row_index;
    double *in_pr,*in_pi,*out_pr,*out_pi;
    mwIndex *in_ir,*in_jc,*out_ir,*out_jc;
   	double thres;

    /* Check for proper number of input and output arguments */    
    if ((nlhs != 1) || (nrhs != 2)){
		mexErrMsgTxt("usage: SPMX = SPARSIFY(MX, THRES).");
    } 
	/* if matrix is complex threshold the norm of the numbers */
	if (mxIsComplex(prhs[0])){
		/* Check data type of input argument  */
		if (mxIsSparse(prhs[0])){

			/* read input */
			in_pr  	= mxGetPr(prhs[0]);
			in_pi  	= mxGetPi(prhs[0]);
			in_ir	= mxGetIr(prhs[0]);
			in_jc	= mxGetJc(prhs[0]);
			nzmax 	= mxGetNzmax(prhs[0]);
			m		= mxGetM(prhs[0]);
			n		= mxGetN(prhs[0]);
			thres 	= mxGetScalar(prhs[1]);

			/* Count new nonzeros */
			newnnz=0;
			for(i=0; i<nzmax; i++){
				if (sqrt(in_pr[i]*in_pr[i] + in_pi[i]*in_pi[i])>thres) {newnnz++;}
			}

			if (newnnz>0){
				/* create output */
				plhs[0] 	= mxCreateSparse(m,n,newnnz,mxCOMPLEX);
				if (plhs[0]==NULL)
					mexErrMsgTxt("Could not allocate enough memory!\n");
				out_pr 		= mxGetPr(plhs[0]);
				out_pi 		= mxGetPr(plhs[0]);
				out_ir 		= mxGetIr(plhs[0]);
				out_jc 		= mxGetJc(plhs[0]);
				passed		= 0;
				out_jc[0]	= 0;
				for (col=0; col<n; col++){
					starting_row_index = in_jc[col];
					stopping_row_index = in_jc[col+1];
					out_jc[col+1] = out_jc[col];
					if (starting_row_index == stopping_row_index)
						continue;
					else {
						for (current_row_index = starting_row_index;
							current_row_index < stopping_row_index;
							current_row_index++)  {
								if (sqrt(in_pr[current_row_index]*in_pr[current_row_index] + 
									     in_pi[current_row_index]*in_pi[current_row_index] ) > thres){

									out_pr[passed]=in_pr[current_row_index];	
									out_pi[passed]=in_pi[current_row_index];	
									out_ir[passed]=in_ir[current_row_index];	
									out_jc[col+1] = out_jc[col+1]+1;
									passed++;
								}
						}
					}
				}
			}
			else{
				plhs[0] = mxCreateSparse(m,n,0,mxCOMPLEX);
			}
		}
		else{ /* for full matrices */
			/* read input */
			in_pr  	= mxGetPr(prhs[0]);
			in_pi  	= mxGetPr(prhs[0]);
			m		= mxGetM(prhs[0]);
			n		= mxGetN(prhs[0]);
			thres 	= mxGetScalar(prhs[1]);

			/* Count new nonzeros */
			newnnz=0;
			for(i=0; i<m*n; i++){
				if (sqrt(in_pr[i]*in_pr[i] + in_pi[i]*in_pi[i])>thres) {newnnz++;}
			}

			if (newnnz>0){
				/* create output */
				plhs[0]	 	= mxCreateSparse(m,n,newnnz,mxCOMPLEX);
				if (plhs[0]==NULL)
					mexErrMsgTxt("Could not allocate enough memory!\n");
				out_pr 		= mxGetPr(plhs[0]);
				out_pi 		= mxGetPi(plhs[0]);
				out_ir 		= mxGetIr(plhs[0]);
				out_jc 		= mxGetJc(plhs[0]);
				passed		= 0;
				out_jc[0] 	= 0;

				for (col=0; col<n; col++){
					out_jc[col+1] = out_jc[col];
					for (current_row_index=0; current_row_index<m; current_row_index++){
							if (sqrt(in_pr[current_row_index+m*col]*in_pr[current_row_index+m*col] +
									 in_pi[current_row_index+m*col]*in_pi[current_row_index+m*col]) > thres){
								
								out_pr[passed]=in_pr[current_row_index+m*col];	
								out_ir[passed]=current_row_index;	
								out_jc[col+1] = out_jc[col+1]+1;
								passed++;
							}
					}
				}
			}
			else{
				plhs[0] = mxCreateSparse(m,n,0,mxCOMPLEX);
			}
		}
	}
	else { 
    	/* Check data type of input argument  */
    	if (mxIsSparse(prhs[0])){

			/* read input */
   			in_pr  	= mxGetPr(prhs[0]);
   			in_ir	= mxGetIr(prhs[0]);
    		in_jc	= mxGetJc(prhs[0]);
	   		nzmax 	= mxGetNzmax(prhs[0]);
			n		= mxGetN(prhs[0]);
			m		= mxGetM(prhs[0]);
			thres 	= mxGetScalar(prhs[1]);

	  		/* Count new nonzeros */
			newnnz=0;
			for(i=0; i<nzmax; i++){
				if ((fabs(in_pr[i]))>thres) {newnnz++;}
			}

			if (newnnz>0){
				/* create output */
	   	 		plhs[0] 	= mxCreateSparse(m,n,newnnz,mxREAL);
				if (plhs[0]==NULL)
					mexErrMsgTxt("Could not allocate enough memory!\n");
   	 			out_pr 		= mxGetPr(plhs[0]);
    			out_ir 		= mxGetIr(plhs[0]);
    			out_jc 		= mxGetJc(plhs[0]);
				passed		= 0;
				out_jc[0]	= 0;
				for (col=0; col<n; col++){
					starting_row_index = in_jc[col];
					stopping_row_index = in_jc[col+1];
					out_jc[col+1] = out_jc[col];
					if (starting_row_index == stopping_row_index)
						continue;
					else {
						for (current_row_index = starting_row_index;
							current_row_index < stopping_row_index;
							current_row_index++)  {
								if (fabs(in_pr[current_row_index])>thres){
									out_pr[passed]=in_pr[current_row_index];	
									out_ir[passed]=in_ir[current_row_index];	
									out_jc[col+1] = out_jc[col+1]+1;
									passed++;
								}
						}
					}
				}
			}
			else{
    			plhs[0] = mxCreateSparse(m,n,0,mxREAL);
			}
		}
		else{ /* for full matrices */
			/* read input */
   			in_pr  	= mxGetPr(prhs[0]);
			n		= mxGetN(prhs[0]);
			m		= mxGetM(prhs[0]);
			thres 	= mxGetScalar(prhs[1]);

	  		/* Count new nonzeros */
			newnnz=0;
			for(i=0; i<m*n; i++){
				if ((fabs(in_pr[i]))>thres) {newnnz++;}
			}

			if (newnnz>0){
				/* create output */
	   	 		plhs[0]	 	= mxCreateSparse(m,n,newnnz,mxREAL);
				if (plhs[0]==NULL)
					mexErrMsgTxt("Could not allocate enough memory!\n");
   	 			out_pr 		= mxGetPr(plhs[0]);
    			out_ir 		= mxGetIr(plhs[0]);
    			out_jc 		= mxGetJc(plhs[0]);
				passed		= 0;
    			out_jc[0] 	= 0;

				for (col=0; col<n; col++){
					out_jc[col+1] = out_jc[col];
					for (current_row_index=0; current_row_index<m; current_row_index++){
							if (fabs(in_pr[current_row_index+m*col])>thres){
								out_pr[passed]=in_pr[current_row_index+m*col];	
								out_ir[passed]=current_row_index;	
								out_jc[col+1] = out_jc[col+1]+1;
								passed++;
							}
					}
				}
			}
			else{
    			plhs[0] = mxCreateSparse(m,n,0,mxREAL);
			}
		}
	}
}
 
