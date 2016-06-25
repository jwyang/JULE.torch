/*================================================================
* function [i,j] = cimgnbmap([nr,nc], nb_r, sample_rate)
*   computes the neighbourhood index matrix of an image,
*   with each neighbourhood sampled.
* Input:
*   [nr,nc] = image size
*   nb_r = neighbourhood radius, could be [r_i,r_j] for i,j
*   sample_rate = sampling rate, default = 1
* Output:
*   [i,j] = each is a column vector, give indices of neighbour pairs
*     UINT32 type
*       i is of total length of valid elements, 0 for first row
*       j is of length nr * nc + 1
*
* See also: imgnbmap.c, id2cind.m
*
* Examples: 
*   [i,j] = imgnbmap(10, 20); % [10,10] are assumed
*
* Stella X. Yu, Nov 12, 2001.

% test sequence:
nr = 15;
nc = 15;
nbr = 1;
[i,j] = cimgnbmap([nr,nc], nbr);
mask = csparse(i,j,ones(length(i),1),nr*nc);
show_dist_w(rand(nr,nc),mask)

*=================================================================*/

# include "mex.h"
# include "math.h"
# include <time.h>

void mexFunction(
    int nargout,
    mxArray *out[],
    int nargin,
    const mxArray *in[]
)
{
    /* declare variables */
    int nr, nc, np, nb, total;
	double *dim, sample_rate;
    int r_i, r_j, a1, a2, b1, b2, self, neighbor;
    int i, j, k, s, t, nsamp, th_rand, no_sample;
    unsigned long *p;
    
    
    /* check argument */
    if (nargin < 2) {
        mexErrMsgTxt("Two input arguments required");
    }
    if (nargout> 2) {
        mexErrMsgTxt("Too many output arguments.");
    }

    /* get image size */
    i = mxGetM(in[0]);
    j = mxGetN(in[0]);
    dim = (double *)mxGetData(in[0]);
    nr = (int)dim[0];
    if (j>1 || i>1) {
        nc = (int)dim[1];
    } else {
        nc = nr;
    }
    np = nr * nc;
    
    /* get neighbourhood size */
    i = mxGetM(in[1]);
    j = mxGetN(in[1]);
    dim = (double*)mxGetData(in[1]);
    r_i = (int)dim[0];
    if (j>1 || i>1) {
        r_j = (int)dim[1];		
    } else {
        r_j = r_i;
    }
    if (r_i<0) { r_i = 0; }
	if (r_j<0) { r_j = 0; }

	/* get sample rate */
	if (nargin==3) {		
		sample_rate = (mxGetM(in[2])==0) ? 1: mxGetScalar(in[2]);
    } else {
		sample_rate = 1;
    }
	/* prepare for random number generator */
	if (sample_rate<1) {
        srand( (unsigned)time( NULL ) );
        th_rand = (int)ceil((double)RAND_MAX * sample_rate);
        no_sample = 0;
    } else {
		sample_rate = 1;
        th_rand = RAND_MAX;
        no_sample = 1;
    }
    
	/* figure out neighbourhood size */

    nb = (r_i + r_i + 1) * (r_j + r_j + 1); 
    if (nb>np) {
        nb = np;
    }
    nb = (int)ceil((double)nb * sample_rate);    

	/* intermediate data structure */
	p = (unsigned long *)mxCalloc(np * (nb+1), sizeof(unsigned long));
	if (p==NULL) {
        mexErrMsgTxt("Not enough space for my computation.");
	}
	
    /* computation */    
	total = 0;
    for (j=0; j<nc; j++) {
    for (i=0; i<nr; i++) {

		self = i + j * nr;

		/* put self in, otherwise the index is not ordered */
		p[self] = p[self] + 1;
		p[self+p[self]*np] = self;

        /* j range */
		b1 = j;
        b2 = j + r_j;
        if (b2>=nc) { b2 = nc-1; }                
    
		/* i range */
        a1 = i - r_i;
		if (a1<0) { a1 = 0; }
        a2 = i + r_i;
        if (a2>=nr) { a2 = nr-1; }
       
		/* number of more samples needed */
		nsamp = nb - p[self];

		k = 0;		
		t = b1;
		s = i + 1;
		if (s>a2) {
			s = a1;
			t = t + 1;
		}
		while (k<nsamp && t<=b2) {
			if (no_sample || (rand()<th_rand)) {
				k = k + 1;
				neighbor = s + t * nr;
				
				p[self] = p[self] + 1;					
				p[self+p[self]*np] = neighbor;
			
				p[neighbor] = p[neighbor] + 1;
				p[neighbor+p[neighbor]*np] = self;
			}
			s = s + 1;
			if (s>a2) {
                s = a1;
				t = t + 1;
			}
		} /* k */

		total = total + p[self];
	} /* i */
    } /* j */
    
    /* i, j */
    out[0] = mxCreateNumericMatrix(total, 1, mxUINT32_CLASS, mxREAL);
	out[1] = mxCreateNumericMatrix(np+1,  1, mxUINT32_CLASS, mxREAL);
    unsigned int *qi = (unsigned int *)mxGetData(out[0]);
	unsigned int *qj = (unsigned int *)mxGetData(out[1]);
	if (out[0]==NULL || out[1]==NULL) {
	    mexErrMsgTxt("Not enough space for the output matrix.");
	}

	total = 0;
    for (j=0; j<np; j++) {
		qj[j] = total;
		s = j + np;
		for (t=0; t<p[j]; t++) {
		    qi[total] = p[s];
			total = total + 1;
			s = s + np;
		}
    }
	qj[np] = total;

	mxFree(p);
}  
