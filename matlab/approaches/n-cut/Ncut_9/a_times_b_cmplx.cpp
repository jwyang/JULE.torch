/*================================================================
a_times_b_cmplx.c = used by a couple of mex functions 
provide Matrix vector multiplications,
and solve triangular systems
(sparse matrix and full vector)

CSC_CmplxVecMult_CAB_double, CSR_CmplxVecMult_CAB_double, 
CSCsymm_CmplxVecMult_CAB_double added by Mirko Visontai (10/24/2003)

*=================================================================*/
# include "math.h"

///*c<-a'*b */
//void scalar_product(
//                 const int m,  const int k, /*nb_rows, nb_columns*/
//                 const double *a,  
//                 const double *b,  
//                 double *c                 
//                 )
//{
//  int i;
//  double d;
//  d = 0;
//    for (i=0;i!=m;i++) 
//    	d+=a[i]*b[i];                 
//  c[0] = d;
//
//}


/*C<-a*A*B+C*/
void CSC_VecMult_CaABC_double(
                 const int m,  const int k, const double alpha,
                 const double *val, const int *indx, 
                 const int *pntrb,
                 const double *b,  
                 double *c)
{
  int i,j,jb,je;

    for (i=0;i!=k;i++){
      jb = pntrb[i];
      je = pntrb[i+1];
      for (j=jb;j!=je;j++)
        c[indx[j]] += alpha * b[i] * val[j];
    }
}

/*C<-a*A'*B+C*/
void CSR_VecMult_CaABC_double(
                 const int k,  const int m, const double alpha,
                 const double *val, const mwIndex *indx, 
                 const mwIndex *pntrb,
                 const double *b,  
                 double *c)
{
  double t;
  const double *pval;
  int i,j,jb,je;

    pval = val;
    for (i=0;i!=m;i++) {
      t = 0;
      jb = pntrb[i];
      je = pntrb[i+1];
      for (j=jb;j!=je;j++)
        t += alpha * b[indx[j]] * (*pval++);
      c[i] += t;
    }
}


/*C<-A*b */
void CSC_VecMult_CAB_double(
                 const int m,  const int k, /*nb_rows, nb_columns*/
                 const double *val, const int *indx, 
                 const int *pntrb,                  
                 const double *b,  
                 double *c                 
                 )
{
  int i,j,jb,je;
  double *pc=c;                                       
    for (i=0;i!=m;i++) *pc++ = 0;                 

    for (i=0;i!=k;i++){
      jb = pntrb[i];
      je = pntrb[i+1];
      for (j=jb;j!=je;j++)
        c[indx[j]] +=  b[i] * val[j];
    }
}

/*C<-A*b (complex)*/
void CSC_CmplxVecMult_CAB_double(
                 const int m,  const int k, 
                 const double *valr, const double *vali,
				 const int *indx, 
                 const int *pntrb,                  
                 const double *br, const double *bi,  
                 double *cr, double *ci                 
                 )
{
  int i,j,jb,je;
  double *pcr=cr;                                       
  double *pci=ci;                                       
    for (i=0;i!=m;i++){
		*pcr++ = 0.0;
		*pci++ = 0.0;
	}

    for (i=0;i!=k;i++){
      jb = pntrb[i];
      je = pntrb[i+1];
      for (j=jb;j!=je;j++){
        cr[indx[j]] +=  (br[i] * valr[j]) - (bi[i] * vali[j]);
        ci[indx[j]] +=  (br[i] * vali[j]) + (bi[i] * valr[j]);
	  }
    }
}

/*C<-A'*b 
 plus rapide que CSC_VecMult_CAB_double */
void CSR_VecMult_CAB_double(
                 const int k,  const int m,
                 const double *val, const int *indx, 
                 const int *pntrb,
                 const double *b,  
                 double *c
                 )
{
  double t;
  const double *pval;
  double *pc=c;                                       
  int i,j,jb,je;
 
    for (i=0;i!=m;i++) *pc++ = 0;                 

    pval = val;
    for (i=0;i!=m;i++) {
      t = 0;
      jb = pntrb[i];
      je = pntrb[i+1];
      for (j=jb;j!=je;j++)
        t +=  b[indx[j]] * (*pval++);
      c[i] += t;
    }
}

/*C<-A'*b (complex) 
 plus rapide que CSC_VecMult_CAB_double */
void CSR_CmplxVecMult_CAB_double(
                 const int k,  const int m, 
                 const double *valr, const double *vali,
				 const int *indx, 
                 const int *pntrb,
                 const double *br, const double *bi,  
                 double *cr, double *ci
                 )
{
  double tr, ti;
  const double *pvalr;
  const double *pvali;
  double *pcr=cr;                                       
  double *pci=ci;                                       
  int i,j,jb,je;
 
    for (i=0;i!=m;i++){
		*pcr++ = 0.0;                 
		*pci++ = 0.0;
	}		

    pvalr = valr;
    pvali = vali;
    for (i=0;i!=m;i++) {
      tr = 0.0;
	  ti = 0.0;
      jb = pntrb[i];
      je = pntrb[i+1];
      for (j=jb;j!=je;j++){
        tr +=  (br[indx[j]] * (*pvalr)) - (bi[indx[j]] * (*pvali));
        ti +=  (br[indx[j]] * (*pvali++)) + (bi[indx[j]] * (*pvalr++));
	  }
      cr[i] += tr;
	  ci[i] += ti;
    }
}



/* C<-A*b (A is symmetric) */
void CSRsymm_VecMult_CAB_double(
                 const int k,  const int m, 
                 const double *val, const mwIndex *indx, 
                 const mwIndex *pntrb,  
                 const double *b,  
                 double *c 
                 )
{
  const double *pval;
  double *pc=c;                                       
  int i,j;
  int jj;
  int rpntrb, rpntre;
  int index, nvals;
  
      
    for (i=0;i!=m;i++) *pc++ = 0;                 
    pval = val;
    for (j=0;j!=k;j++){
      rpntrb = pntrb[j];
      rpntre = pntrb[j+1];
      for (jj=rpntrb;jj!=rpntre;jj++) {
        index = indx[jj];
        if ( index == j ) {
          c[j] +=  b[j] * (*pval++);
          continue;
        }
		if ( index > j ) {
       		c[index] +=  b[j] * (*pval);

       		c[j] +=  b[index] * (*pval++);
		}
		else {
			pval++;
		}
      }
    }
}


/* C<-A*b (A is symmetric and complex) */
void CSRsymm_CmplxVecMult_CAB_double(
                 const int k,  const int m, 
                 const double *valr, const double *vali,
				 const int *indx, 
                 const int *pntrb,  
                 const double *br, const double *bi,  
                 double *cr, double *ci
                 )
{
  const double *pvalr, *pvali;
  double *pcr=cr;                                       
  double *pci=ci;                                       
  int i,j;
  int jj;
  int rpntrb, rpntre;
  int index, nvals;
  
      
    for (i=0;i!=m;i++){
		*pcr++ = 0.0;                 
		*pci++ = 0.0;                 
	}

    pvalr = valr;
    pvali = vali;
    for (j=0;j!=k;j++){
      rpntrb = pntrb[j];
      rpntre = pntrb[j+1];
      for (jj=rpntrb;jj!=rpntre;jj++) {
        index = indx[jj];
        if ( index == j ) {
          cr[j] +=  (br[j] * (*pvalr)) - (bi[j] * (*pvali));
          ci[j] +=  (br[j] * (*pvali++)) + (bi[j] * (*pvalr++));
          continue;
        }
       	if ( index > j ) {
        	cr[index] +=  (br[j] * (*pvalr)) - (bi[j] * (*pvali));
        	ci[index] +=  (br[j] * (*pvali)) + (bi[j] * (*pvalr));
        
        	cr[j] +=  (br[index] * (*pvalr)) - (bi[index] * (*pvali));
        	ci[j] +=  (br[index] * (*pvali++)) + (bi[index] * (*pvalr++));
		}
		else {
			pvalr++;
			pvali++;
		}
        
      }
    }
}


/*C<-A\B; with Lower triangular A*/
void CSC_VecTriangSlvLD_CAB_double( 
                 const int m,  
                 const double *val,
                 const int *indx, const int *pntrb, 
                 const double *b,  
                 double *c)
{
  int i, j, jb, je;
  double *pc=c;
  double z; 

    for (i=0;i!=m;i++){
      *pc = b[i];
      pc++;
    }                                     

    pc=c;
    for (i=0;i!=m;i++) {
      jb = pntrb[i];
      je = pntrb[i+1];
      z =  pc[i] / val[jb];
      pc[i] = z;
      for (j=jb+1;j<je;j++) {
        c[indx[j]] -= z*val[j];
      }
    }
}

/*C<-A\B; with Upper triangular A*/
void CSC_VecTriangSlvUD_CAB_double(
                 const int m,  
                   const double *val,
                 const int *indx, const int *pntrb,
                 const double *b,  
                 double *c)
{
  int i, j, jb, je, index;
  double *pc=c;
  double z; 

    for (i=0;i!=m;i++){
      *pc = b[i];
      pc++;
    }                                     

    pc=c;
    for (i=m-1;i!=-1;i--) {
      jb = pntrb[i];
      je = pntrb[i+1]-1;
      z = pc[i] /val[je];
      pc[i] = z;
      for (j=jb;j<je;j++) {
          c[indx[j]] -= z * val[j];
      }
    }
}
/*C<-A'\B; where A is upper (little slower than CSC)*/
void CSR_VecTriangSlvLD_CAB_double(
                 const int m,   
                 const double *val, 
                 const int *indx, const int *pntrb,
                 const double *b,   
                 double *c)
{
  int i, j, jb, je, index;
  double *pc=c;
  double z; 
  double valtmp;

    pc=c;
    for (i=0;i!=m;i++) {
      z = 0;
      jb = pntrb[i];
      je = pntrb[i+1];
      for (j=jb;j<je;j++) {
        index = indx[j];
        if ( index == i ) {
          valtmp = val[j];
        } else {
          z += c[index] * val[j];
        }
      }
      pc[i] = (b[i] - z) / valtmp;
    }
}

/*C<-A'\B; where A is lower  (little slower than CSC)*/
void CSR_VecTriangSlvUD_CAB_double(
                 const int m,   
                   const double *val, 
                 const int *indx, const int *pntrb, 
                 const double *b,   
                 double *c)
{
  int i, j, jb, je, index;
  double *pc=c;
  double valtmp;
  double z; 

    pc=c;
    for (i=m-1;i!=-1; i--) {
      z = 0;
      jb = pntrb[i];
      je =  pntrb[i+1];
      for (j=jb+1; j<je; j++) {
        z += c[indx[j]] * val[j];
      }
      pc[i] = (b[i] - z) / val[jb];
    }
}

/*C<-A*B, where A is (m,k) and B is (k,n)*/
void CSC_MatMult_CAB_double(
                 const int m, const int n, const int k, 
                 const double *val, const int *indx, 
                 const int *pntrb,
                 const double *b, const int ldb, 
                 double *c, const int ldc)
{
  int i,j,jb,je;
  double *pc=c;                                       
  int l;                                               

  for (l=0;l!=n;l++)                           
    for (i=0;i!=m;i++) *pc++ = 0;                 

  for (l=0;l!=n;l++) {                                 
    for (i=0;i!=k;i++){
      jb = pntrb[i];
      je = pntrb[i+1];
      for (j=jb;j!=je;j++)
        c[indx[j]] +=  b[i] * val[j];
    }
    /*c += ldc; b += ldb;  */                              
    c += m; b += m;                                
  }                                                    
}
