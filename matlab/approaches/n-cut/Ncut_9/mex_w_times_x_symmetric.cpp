/*================================================================
* mex_w_times_x_symmetric.c = used by ncuthard2.m in eigensolver.
*
* Examples: 
*     mex_w_times_x_c_symmetric(x,triu(A)) = A*x;     
*     A is sparse and symmetric, but x is full
*
* Timothee Cour, Oct 12, 2003.

% test sequence:
    m=100;
    n=50;    
    x=rand(n,1);
    A=sprand(m,n,0.01);    

    y2 = mex_w_times_x_c_symmetric(x,triu(A));
    y1=A*x;
    max(abs(y1-y2))
*=================================================================*/

# include "math.h"
# include "mex.h"
# include "a_times_b_cmplx.cpp"
/*# include "a_times_b.c"*/


void mexFunction(
    int nargout,
    mxArray *out[],
    int nargin,
    const mxArray *in[]
)
{
    int np, nc;
    mwIndex*ir, *jc;
    double *x, *y, *pr;
    
    if (nargin < 2) {//voir
        mexErrMsgTxt("Four input arguments required !");
    }
    if (nargout>1) {
        mexErrMsgTxt("Too many output arguments.");
    }
        
	x = mxGetPr(in[0]);
        pr = mxGetPr(in[1]);
        ir = mxGetIr(in[1]);
        jc = mxGetJc(in[1]);
   
    	np = mxGetM(in[1]);
    	nc = mxGetN(in[1]);
    	    
    	out[0] = mxCreateDoubleMatrix(np,1,mxREAL);
	y = mxGetPr(out[0]);
	
	CSRsymm_VecMult_CAB_double(np,nc,pr,ir,jc,x,y); 
} 
