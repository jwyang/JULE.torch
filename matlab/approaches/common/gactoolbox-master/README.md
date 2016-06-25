GACToolbox
==========

Graph Agglomerative Clustering (GAC) toolbox

Introduction
------------

Gactoolbox is a summary of our research of agglomerative clustering on a graph. Agglomerative clustering, which iteratively merges small clusters, is commonly used for clustering because it is conceptually simple and produces a hierarchy of clusters. Classifical aggolomerative clustering algorithms, such as average linkage and DBSCAN, were widely used in many areas. Those algorithms, however, are not designed for clustering on a graph. This toolbox implements the following algorithms for agglomerative clustering on a directly graph.

1. Structural descriptor based algorithms (`gacCluster.m`). We define a cluster descriptor based on the graph structure, and each merging is determined by maximizes the increment of the descriptor. Two descriptors, including zeta function and path integral, are implemented. You can also design new descriptor (creating functions similar to `gacPathEntropy.m` and `gacPathCondEntropy.m`) and develop new algorithms with our code.

2. Graph degree linkage (`gdlCluster.m`). It is a simple and effective algorithm, with better performance than normalized cuts and spectral clustering, and is faster.

This toolbox is written and maintained by Wei Zhang (`wzhang009 at gmail.com`).
Please send me an email if you find any bugs or have any suggestions.

Examples
--------
Preparations:

1. Compile mex functions
2. Add 'gacfiles' and 'gdlfiles' to your matlab paths
3. Calculate a pairwise distance matrix from your data

```matlab
K = 20;
a = 1;
z = 0.01;

% path integral

clusteredLabels = gacCluster (distance_matrix, groupNumber, 'path', K, a, z);

% zeta function

clusteredLabels = gacCluster (distance_matrix, groupNumber, 'zeta', K, a, z);

% GDL-U algorithm

clusteredLabels = gdlCluster(distance_matrix, groupNumber, K, a, false);

% AGDL algorithm

clusteredLabels = gdlCluster(distance_matrix, groupNumber, K, a, true);
```

Citations
---------

Please cite the following papers, if you find the code is helpful.
 
* W. Zhang, D. Zhao, and X. Wang. 
Agglomerative clustering via maximum incremental path integral.
Pattern Recognition, 46 (11): 3056-3065, 2013.

* W. Zhang, X. Wang, D. Zhao, and X. Tang. 
Graph Degree Linkage: Agglomerative Clustering on a Directed Graph.
in Proceedings of European Conference on Computer Vision (ECCV), 2012.

Additional Notes
----------------

1. How to compile mex files?

   I include mexw64 files. If you use a system other than win64, you can find a file called compileMex.m to help you build the mex files.

2. We provide MATLAB implementation of structural descriptor based clustering and MATLAB-C++ mixed implementation of graph degree linkage. The MATLAB implementation is for ease of understanding, although it's inefficient. In the future we will add MATLAB implementation of graph degree linkage.

   In speed: AGDL > GDL-U > path integral > zeta function

3. GDL-U and AGDL have similar performance. GDL-U is for small datasets and AGDL is for large datasets. 

   AGDL has an additional parameter Kc in gdlMergingKNN_c.m. The larger Kc is, the closer performance AGDL has to GDL-U and slower the algorithm is. Default Kc = 10 is a good trade-off for most datasets.
