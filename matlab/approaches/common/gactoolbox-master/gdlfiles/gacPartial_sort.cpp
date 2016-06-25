
// use C++ partial_sort for partial sort

#include "mex.h"
#include "gacPartial_sort.h"

// Determine type of data, run
void run_partial_sort(mxArray *plhs[], const mxArray *inarr, int rank, int ncols, int nrows, const int dim) {
    void *indata = mxGetData(inarr);
    void *outdata = mxGetData(plhs[0]);
    if (dim == 1) {
        partial_sort_cols((double *)outdata, (double *)indata, rank, ncols, nrows);
    }
    else {
        partial_sort_rows((double *)outdata, (double *)indata, rank, ncols, nrows);
    }
}

void run_partial_sort_withIdx (mxArray *plhs[], const mxArray *inarr, int rank, int ncols, int nrows, const int dim) {
    void *indata = mxGetData(inarr);
    void *outdata = mxGetData(plhs[0]);
    double *pIdx = mxGetPr(plhs[1]);
    if (dim == 1) {
        partial_sort_cols_withIdx((double *)outdata, pIdx, (double *)indata, rank, ncols, nrows);
    }
    else {
        partial_sort_rows_withIdx((double *)outdata, pIdx, (double *)indata, rank, ncols, nrows);
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check inputs
    if (nrhs != 3) {
        mexErrMsgTxt("Arguments should be the matrix of columns and the rank of the desired element.");
    }
    if (!mxIsNumeric(prhs[0])) {
        mexErrMsgTxt("First argument must be a numeric matrix.");
    }
    if (!mxIsNumeric(prhs[1]) || mxGetNumberOfDimensions(prhs[1]) > 2 || mxGetM(prhs[1]) != 1 || mxGetN(prhs[1]) != 1) {
        mexErrMsgTxt("Second argument must be a scalar.");
    }
    if (!mxIsNumeric(prhs[2]) || mxGetNumberOfDimensions(prhs[2]) > 2 || mxGetM(prhs[2]) != 1 || mxGetN(prhs[2]) != 1) {
        mexErrMsgTxt("Third argument must be a scalar.");
    }

    const int nrows = mxGetM(prhs[0]);
    const int ncols = mxGetN(prhs[0]);
    int rank = (int) (mxGetScalar(prhs[1]) + 0.5);
    int dim = (int) (mxGetScalar(prhs[2]) + 0.5);
    // Validate rank argument
    if (rank < 1 || (dim == 1 && rank > nrows) || (dim == 2 && rank > ncols) ) {
        mexErrMsgTxt("Rank is out of the range.");
    }
    if (dim != 1 && dim != 2) {
        mexErrMsgTxt("Dim should be 1 or 2.");
    }

    double *pIdx = NULL;
    if (dim == 1) {
        plhs[0] = mxCreateDoubleMatrix(rank, ncols, mxREAL);
    }
    else {
        plhs[0] = mxCreateDoubleMatrix(nrows, rank, mxREAL);
    }
    switch (nlhs) {
        case 1:
            run_partial_sort(plhs, prhs[0], rank, ncols, nrows, dim);
            break;
        case 2:
            if (dim == 1) {
                plhs[1] = mxCreateDoubleMatrix(rank, ncols, mxREAL);
                memset((void*)mxGetPr(plhs[1]), 0, rank*ncols*sizeof(double));
            }
            else {
                plhs[1] = mxCreateDoubleMatrix(nrows, rank, mxREAL);
                memset((void*)mxGetPr(plhs[1]), 0, rank*nrows*sizeof(double));
            }
            run_partial_sort_withIdx(plhs, prhs[0], rank, ncols, nrows, dim);
            break;
        default:
            mexErrMsgTxt("too many output!");
    }
}