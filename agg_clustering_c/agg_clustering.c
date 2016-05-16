#include <lua.h>                               /* Always include this */
#include <lauxlib.h>                           /* Always include this */
#include <lualib.h>                            /* Always include this */
#include <luaT.h>                              /* Always include this */
#include <TH.h>                                /* Always include this */
#include <omp.h>

#define Real Float
#define real float
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)\

/***************************************/
/****** Compute Cluster Affinity *******/ 
/***************************************/
static int icompute_CAff(lua_State *L) {  // Cluster Affinity
	// inputs:
	//  1: affinity matrix for samples
	//  2: knn matrix for clusters
	//  3: indice for clusters
	int i, j;
	int m, n;
	float A_c_i_j, A_c_j_i, s_W_c_j_i, s_W_c_i_j;

    THTensor * W   = luaT_checkudata(L, 1, torch_Tensor);  // get sample affinity matrix
    THTensor * NNs = luaT_checkudata(L, 2, torch_Tensor);  // get sample affinity matrix
    THTensor * Y   = luaT_checkudata(L, 3, torch_Tensor);  // get sample affinity matrix

    // printf("W Size: %ld %ld\n", W->size[0], W->size[1]);    
    // printf("NNs Size: %ld %ld\n", NNs->size[0], NNs->size[1]);  
    // printf("Y Size: %ld %ld\n", Y->size[0], Y->size[1]);      
    
    THTensor * A_s = THTensor_(new)();
    THTensor_(resizeAs)(A_s, NNs);

    THTensor * A_us = THTensor_(new)();
    THTensor_(resizeAs)(A_us, NNs);

    int nclusters = NNs->size[0];    
    for (i = 0; i < nclusters; ++i) {    	
    	#pragma omp parallel for private(s_W_c_j_i, s_W_c_i_j, j, m, n, A_c_i_j, A_c_j_i)
    	for (j =  i; j < nclusters; ++j) {

    		if (THTensor_(get2d)(NNs, i, j) == 0 && THTensor_(get2d)(NNs, j, i) == 0) {
    			THTensor_(set2d)(A_us, j, i, 0);
    			THTensor_(set2d)(A_us, i, j, 0);
    			THTensor_(set2d)(A_s, j, i, 0);
    			THTensor_(set2d)(A_s, i, j, 0);
    			continue;
    		}

    		if (i == j) {
    			THTensor_(set2d)(A_us, j, i, 0);
    			THTensor_(set2d)(A_s, j, i, 0);
    			continue;
    		}

            // get the size of Y[i] and Y[j]
            int Y_i_size, Y_j_size;
            Y_i_size = 0;
            Y_j_size = 0;
			for (m = 0; m < Y->size[1]; ++m) {
				if (THTensor_(get2d)(Y, i, m) != 0) { // reach the end of list
					++Y_i_size;
				}
				if (THTensor_(get2d)(Y, j, m) != 0) { // reach the end of list
					++Y_j_size;
				}
			}

    		// compute affinity from cluster i to cluster j
			A_c_i_j = 0;
			// #pragma omp parallel for private(s_W_c_j_i, s_W_c_i_j, m, n) shared(A_c_i_j)
			for (m = 0; m < Y_i_size; ++m) {
				s_W_c_j_i = 0;
				s_W_c_i_j = 0;
				for (n = 0; n < Y_j_size; ++n) {
					// printf("%d %d\n", THTensor_(get2d)(Y, j, n), THTensor_(get2d)(Y, i, m));
					s_W_c_j_i += THTensor_(get2d)(W, THTensor_(get2d)(Y, j, n) - 1, THTensor_(get2d)(Y, i, m) - 1);
					s_W_c_i_j += THTensor_(get2d)(W, THTensor_(get2d)(Y, i, m) - 1, THTensor_(get2d)(Y, j, n) - 1);
					// W_samples.at<float>(label_indice[j][n], label_indice[i][m]);
				}
				A_c_i_j += s_W_c_j_i * s_W_c_i_j;
			}

			// compute affinity from cluster j to cluster i
			A_c_j_i = 0;
			// #pragma omp parallel for private(s_W_c_j_i, s_W_c_i_j, m, n) shared(A_c_j_i)
			for (m = 0; m < Y_j_size; ++m) {
				s_W_c_j_i = 0;
				s_W_c_i_j = 0;
				for (n = 0; n < Y_i_size; ++n) {
					// printf("%d %d\n", THTensor_(get2d)(Y, j, n), THTensor_(get2d)(Y, i, m));
					s_W_c_j_i += THTensor_(get2d)(W, THTensor_(get2d)(Y, j, m) - 1, THTensor_(get2d)(Y, i, n) - 1);
					s_W_c_i_j += THTensor_(get2d)(W, THTensor_(get2d)(Y, i, n) - 1, THTensor_(get2d)(Y, j, m) - 1);
					// W_samples.at<float>(label_indice[j][n], label_indice[i][m]);
				}
				A_c_j_i += s_W_c_i_j * s_W_c_j_i;
			}

			THTensor_(set2d)(A_us, j, i, A_c_i_j);
			THTensor_(set2d)(A_us, i, j, A_c_j_i);
			THTensor_(set2d)(A_s, i, j, A_c_i_j / (Y_j_size * Y_j_size)  + A_c_j_i / (Y_i_size * Y_i_size));
			THTensor_(set2d)(A_s, j, i, 0);
    	}
    }
  	luaT_pushudata(L, A_us, torch_Tensor);
  	luaT_pushudata(L, A_s, torch_Tensor);
  	// lua_pushtensor(L, A_s);
    return 2;
}

/* Register this file's functions with the
 * luaopen_libraryname() function, where libraryname
 * is the name of the compiled .so output. In other words
 * it's the filename (but not extension) after the -o
 * in the cc command.
 *
 * So for instance, if your cc command has -o power.so then
 * this function would be called luaopen_power().
 *
 * This function should contain lua_register() commands for
 * each function you want available from Lua.
 *
*/
int luaopen_agg_clustering(lua_State *L){
	lua_register(L, "compute_CAff", icompute_CAff);
	return 0;
}
