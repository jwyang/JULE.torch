function [L_i, L_j] = gdlAffinity (graphW, cluster_i, cluster_j)
%% Compute conditional complexity from the subpart of the weighted adjacency matrix
% Inputs:
%   - graphW: the matrix P
%	- cluster_i: index vector of cluster i
%	- cluster_j: index vector of cluster j
% Output:
%	- L_ij - the sum of conditional complexities of cluster i and j after merging.
% by Wei Zhang (wzhang009 at gmail.com), Nov., 7, 2011
%
% This file is the matlab version of gdlAffinity_c.cpp, to help you understand the code
%
% j1 -> i -> j2, j2 -> i -> j1

num_i = numel(cluster_i);
num_j = numel(cluster_j);

% detecting cross elements (this check costs much and is unnecessary)
% if length(unique([cluster_i(:); cluster_j(:)])) ~= (num_i + num_j), error('GAC: two clusters have overlaps!'); end

L_ij = graphW(cluster_i, cluster_j);
L_ji = graphW(cluster_j, cluster_i);
L_i = sum(L_ij,1)*sum(L_ji,2) / (num_i*num_i);
L_j = sum(L_ji,1)*sum(L_ij,2) / (num_j*num_j);
% L_ij = L_i + L_j;

end
