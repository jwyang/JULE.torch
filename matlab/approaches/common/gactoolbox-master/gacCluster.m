function clusteredLabels = gacCluster (distance_matrix, groupNumber, strDescr, K, a, z)
%% Graph Agglomerative Clustering toolbox
% Input: 
%   - distance_matrix: pairwise distances, d_{i -> j}
%   - groupNumber: the final number of clusters
%   - strDescr: structural descriptor. The choice can be
%                 - 'zeta':  zeta function based descriptor
%                 - 'path':  path integral based descriptor
%   - K: the number of nearest neighbors for KNN graph, default: 20
%   - p: merging (p+1)-links in l-links algorithm, default: 1
%   - a: for covariance estimation, default: 1 
%       sigma^2 = (\sum_{i=1}^n \sum_{j \in N_i^K} d_{ij}^2) * a
%   - z: (I - z*P), default: 0.01
% Output:
%   - clusteredLabels: clustering results
% by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011
%
% Please cite the following papers, if you find the code is helpful
% 
% W. Zhang, D. Zhao, and X. Wang. 
% Agglomerative clustering via maximum incremental path integral.
% Pattern Recognition, 46 (11): 3056-3065, 2013.
%
% W. Zhang, X. Wang, D. Zhao, and X. Tang. 
% Graph Degree Linkage: Agglomerative Clustering on a Directed Graph.
% in Proceedings of European Conference on Computer Vision (ECCV), 2012.

%% parse inputs
disp('--------------- Graph Structural Agglomerative Clustering ---------------------');

if nargin < 2, error('GAC: input arguments are not enough!'); end
if nargin < 3, strDescr = 'path'; end
if nargin < 4, K = 20;   end
if nargin < 5, a = 1; end
if nargin < 6, z = 0.01; end

%% initialization

disp('---------- Building graph and forming initial clusters with l-links ---------');
[graphW, NNIndex] = gacBuildDigraph(distance_matrix, K, a);
% from adjacency matrix to probability transition matrix
graphW = bsxfun(@times, 1./sum(graphW,2), graphW); % row sum is 1
initialClusters = gacNNMerge(distance_matrix, NNIndex);
clear distance_matrix NNIndex

disp('-------------------------- Zeta merging --------------------------');
clusteredLabels = gacMerging(graphW, initialClusters, groupNumber, strDescr, z);

end