function L_ij = gacPathCondEntropy (IminuszW, cluster_i, cluster_j)
%% Compute conditional complexity from the subpart of the weighted adjacency matrix
% Inputs:
%   - IminuszW: the matrix (I - z*P)
%	- cluster_i: index vector of cluster i
%	- cluster_j: index vector of cluster j
% Output:
%	- L_ij - the sum of conditional complexities of cluster i and j after merging.
% by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

num_i = numel(cluster_i);
num_j = numel(cluster_j);

% detecting cross elements (this check costs much and is unnecessary)
% if length(unique([cluster_i(:); cluster_j(:)])) ~= (num_i + num_j), error('GAC: two clusters have overlaps!'); end

ijGroupIndex = [cluster_i(:); cluster_j(:)];

y_ij = zeros(num_i+num_j,2);  % [y_i, y_j]
y_ij(1:num_i,1) = 1;
y_ij(num_i+1:end,2) = 1;
% y_ij(num_i+1:num_i+num_j+num_i) = 0;
% compute the coditional complexity of cluster i and j
L_ij = IminuszW(ijGroupIndex, ijGroupIndex) \ y_ij;
L_ij = sum(L_ij(1:num_i,1)) / (num_i*num_i) + sum(L_ij(num_i+1:end,2)) / (num_j*num_j);

end
