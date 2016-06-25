/*================================================================
* spmtimesd.c
* This routine computes a sparse matrix times a diagonal matrix
* whose diagonal entries are stored in a full vector.
*
* Examples: 
*     spmtimesd(m,d,[]) = diag(d) * m,
*     spmtimesd(m,[],d) = m * diag(d)
*     spmtimesd(m,d1,d2) = diag(d1) * m * diag(d2)
*     m could be complex, but d is assumed real
*
* Stella X. Yu's first MEX function, Nov 9, 2001.

% test sequence:

m = 1000;
n = 2000;
a=sparse(rand(m,n));
d1 = rand(m,1);
d2 = rand(n,1);
tic; b=spmtimesd(a,d1,d2); toc
tic; bb = spdiags(d1,0,m,m) * a * spdiags(d2,0,n,n); toc
e = (bb-b);
max(abs(e(:)))

*=================================================================*/

# include "mex.h"

void mexFunction(
    int nargout,
    mxArray *out[],
    int nargin,
    const mxArray *in[]
)
{
    /* declare variables */
    int i, j, k, m, n, nzmax, xm, yn;
    mwIndex *pir, *pjc, *qir, *qjc;
    double *x, *y, *pr, *pi, *qr, *qi;
    
    /* check argument */
    if (nargin != 3) {
        mexErrMsgTxt("Three input arguments required");
    }
    if (nargout>1) {
        mexErrMsgTxt("Too many output arguments.");
    }
    if (!(mxIsSparse(in[0]))) {
        mexErrMsgTxt("Input argument #1 must be of type sparse");
    }
    if ( mxIsSparse(in[1]) || mxIsSparse(in[2]) ) {
        mexErrMsgTxt("Input argument #2 & #3 must be of type full");
    }
  
    /* computation starts */
    m = mxGetM(in[0]);
    n = mxGetN(in[0]);
    pr = mxGetPr(in[0]);
    pi = mxGetPi(in[0]);
    pir = mxGetIr(in[0]);
    pjc = mxGetJc(in[0]);
 
    i = mxGetM(in[1]); 
    j = mxGetN(in[1]);
    xm = ((i>j)? i: j);

    i = mxGetM(in[2]); 
    j = mxGetN(in[2]);
    yn = ((i>j)? i: j);
   
    if ( xm>0 && xm != m) {
        mexErrMsgTxt("Row multiplication dimension mismatch.");
    }
    if ( yn>0 && yn != n) {
        mexErrMsgTxt("Column multiplication dimension mismatch.");
    }
    
 
    nzmax = mxGetNzmax(in[0]);
    mxComplexity cmplx = (pi==NULL ? mxREAL : mxCOMPLEX);    
    out[0] = mxCreateSparse(m,n,nzmax,cmplx);
    if (out[0]==NULL) {
        mexErrMsgTxt("Not enough space for the output matrix.");
    }    
   
    qr = mxGetPr(out[0]);
    qi = mxGetPi(out[0]);
    qir = mxGetIr(out[0]);
    qjc = mxGetJc(out[0]);
    
    /* left multiplication */
    x = mxGetPr(in[1]);
    if (yn==0) {
        for (j=0; j<n; j++) {
            qjc[j] = pjc[j];
            for (k=pjc[j]; k<pjc[j+1]; k++) {
                i = pir[k];   
                qir[k] = i;
                 qr[k] = x[i] * pr[k];
                 if (cmplx) {
                     qi[k] = x[i] * pi[k];
                 }
            }
        }
        qjc[n] = k;
        return;
    }
    
    /* right multiplication */
    y = mxGetPr(in[2]);
    if (xm==0) {
        for (j=0; j<n; j++) {
            qjc[j] = pjc[j];
            for (k=pjc[j]; k<pjc[j+1]; k++) {
                qir[k] = pir[k];
                qr[k]  = pr[k] * y[j];
                if (cmplx) {
                    qi[k] = qi[k] * y[j];
                }
            }
       }
       qjc[n] = k;
       return;
   }
    
   /* both sides */
   for (j=0; j<n; j++) {
       qjc[j] = pjc[j];
       for (k=pjc[j]; k<pjc[j+1]; k++) {
           i = pir[k];
           qir[k]= i;
           qr[k] = x[i] * pr[k] * y[j];
           if (cmplx) {
               qi[k] = x[i] * qi[k] * y[j];      
           }
       }
       qjc[n] = k;
   }
   
}   
