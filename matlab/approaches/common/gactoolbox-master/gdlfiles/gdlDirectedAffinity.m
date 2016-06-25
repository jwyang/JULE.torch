function L_i = gdlDirectedAffinity (graphW, cluster_i, cluster_j)
%% Compute conditional complexity from the subpart of the weighted adjacency matrix
% Inputs:
%   - graphW: the matrix P
%	- cluster_i: index vector of cluster i
%	- cluster_j: index vector of cluster j
% Output:
%	- L_ij - the sum of conditional complexities of cluster i and j after merging.
% by Wei Zhang (wzhang009 at gmail.com), Nov., 7, 2011
%%%%%%% this function is replaced by gacPathAffinity_fast_c (MEX-file)
%%%%%%% this file is kept for easy reading
% j1 -> i -> j2, j2 -> i -> j1

% num_i = numel(cluster_i);
% num_j = numel(cluster_j);

L_i = sum(graphW(cluster_i, cluster_j),1)*sum(graphW(cluster_j, cluster_i),2); % / (num_i*num_i);

end
