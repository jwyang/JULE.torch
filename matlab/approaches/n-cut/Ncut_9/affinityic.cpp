/*================================================================
* function w = affinityic(emag,ephase,pi,pj,sigma)
* Input:
*   emag = edge strength at each pixel
*   ephase = edge phase at each pixel
*   [pi,pj] = index pair representation for MALTAB sparse matrices
*   sigma = sigma for IC energy
* Output:
*   w = affinity with IC at [pi,pj]
*

% test sequence
f = synimg(10);
[i,j] = cimgnbmap(size(f),2);
[ex,ey,egx,egy] = quadedgep(f);
a = affinityic(ex,ey,egx,egy,i,j)
show_dist_w(f,a);

* Stella X. Yu, Nov 19, 2001.
*=================================================================*/

# include "mex.h"
# include "math.h"

void mexFunction(
    int nargout,
    mxArray *out[],
    int nargin,
    const mxArray *in[]
)
{
    /* declare variables */
    int nr, nc, np, total;
    int i, j, k, ix, iy, jx, jy, ii, jj, iip1, jjp1, iip2, jjp2, step;
    double sigma, di, dj, a, z, maxori, phase1, phase2, slope;
// 	int *ir, *jc;
    mwIndex*ir,*jc;
	unsigned int *pi, *pj;
	double *emag, *ephase, *w;
    
    /* check argument */
    if (nargin<4) {
        mexErrMsgTxt("Four input arguments required");
    }
    if (nargout>1) {
        mexErrMsgTxt("Too many output arguments");
    }

    /* get edgel information */
	nr = mxGetM(in[0]);
	nc = mxGetN(in[0]);
	if ( nr*nc ==0 || nr != mxGetM(in[1]) || nc != mxGetN(in[1]) ) {
	    mexErrMsgTxt("Edge magnitude and phase shall be of the same image size");
	}
    emag = mxGetPr(in[0]);
    ephase = mxGetPr(in[1]);
    np = nr * nc;
    
    /* get new index pair */
    if (!mxIsUint32(in[2]) | !mxIsUint32(in[3])) {
        mexErrMsgTxt("Index pair shall be of type UINT32");
    }
    if (mxGetM(in[3]) * mxGetN(in[3]) != np + 1) {
        mexErrMsgTxt("Wrong index representation");
    }
    pi = (unsigned int*)mxGetData(in[2]);
    pj = (unsigned int*)mxGetData(in[3]);    

    /* create output */
    out[0] = mxCreateSparse(np,np,pj[np],mxREAL);
    if (out[0]==NULL) {
	    mexErrMsgTxt("Not enough memory for the output matrix");
	}
	w = mxGetPr(out[0]);
	ir = mxGetIr(out[0]);
	jc = mxGetJc(out[0]);
	
    /* find my sigma */
	if (nargin<5) {
	    sigma = 0;
    	for (k=0; k<np; k++) { 
    	    if (emag[k]>sigma) { sigma = emag[k]; }
    	}
    	sigma = sigma / 6;
    	printf("sigma = %6.5f",sigma);
	} else {
	    sigma = mxGetScalar(in[4]);
	}
	a = 0.5 / (sigma * sigma);
	
    /* computation */ 
    total = 0;
    for (j=0; j<np; j++) {            

        jc[j] = total;
        jx = j / nr; /* col */
        jy = j % nr; /* row */
        
        for (k=pj[j]; k<pj[j+1]; k++) {
        
            i = pi[k];
          
            if (i==j) {
                maxori = 1;
            
            } else {
        
                ix = i / nr; 
                iy = i % nr;

                /* scan */            
                di = (double) (iy - jy);
                dj = (double) (ix - jx);
            
                maxori = 0.;
	            phase1 = ephase[j];

	               
                /* sample in i direction */
                if (abs(di) >= abs(dj)) {  
            	    slope = dj / di;
            	    step = (iy>=jy) ? 1 : -1;
            	
              	    iip1 = jy;
            	    jjp1 = jx;
	
	               
	                for (ii=0;ii<abs(di);ii++){
	                    iip2 = iip1 + step;
	                    jjp2 = (int)(0.5 + slope*(iip2-jy) + jx);
	  	  
	                    phase2 = ephase[iip2+jjp2*nr];
               
	                    if (phase1 != phase2) {
	                        z = (emag[iip1+jjp1*nr] + emag[iip2+jjp2*nr]);
	                        if (z > maxori){
	                            maxori = z;
	                        }
	                    } 
	             
	                    iip1 = iip2;
	                    jjp1 = jjp2;
	                    phase1 = phase2;
	                }
	            
	            /* sample in j direction */    
                } else { 
	                slope = di / dj;
	                step =  (ix>=jx) ? 1: -1;

    	            jjp1 = jx;
	                iip1 = jy;	           
	    
	 
	                for (jj=0;jj<abs(dj);jj++){
	                    jjp2 = jjp1 + step;
	                    iip2 = (int)(0.5+ slope*(jjp2-jx) + jy);
	  	  
	                    phase2 = ephase[iip2+jjp2*nr];
	     
	                    if (phase1 != phase2){
	                        z = (emag[iip1+jjp1*nr] + emag[iip2+jjp2*nr]);
	                        if (z > maxori){ 
	                            maxori = z; 
	                        }
	                        
	                    }
	  
	                    iip1 = iip2;
	                    jjp1 = jjp2;
	                    phase1 = phase2;
	                }
                }            
            
                maxori = 0.5 * maxori;
                maxori = exp(-maxori * maxori * a);
            }       
		    ir[total] = i;

		    w[total] = maxori;
		    total = total + 1;
			
		} /* i */
    } /* j */
        
    jc[np] = total;
}  
