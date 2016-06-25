function L_ij = gacZetaCondEntropy (IminuszW, cluster_i, cluster_j)
%% Compute conditional complexity from the subpart of the weighted adjacency matrix
% Inputs:
%   - IminuszW: the matrix (I - z*P)
%	- cluster_i: index vector of cluster i
%	- cluster_j: index vector of cluster j
% Output:
%	- L_ij - the sum of conditional complexities of cluster i and j after merging.
% by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

num_i = length(cluster_i);
num_j = length(cluster_j);

% detecting cross elements (this check costs much and is unnecessary)
% if length(unique([cluster_i(:); cluster_j(:)])) ~= (num_i + num_j), error('GAC: two clusters have overlaps!'); end

ijGroupIndex = [cluster_i(:); cluster_j(:)];

% compute the corresponding self-similaries
logZetaSelfSim = log(real(diag(inv(IminuszW(ijGroupIndex, ijGroupIndex)))));
% compute the coditional complexity of cluster i and j
L_ij = sum(logZetaSelfSim(1:num_i))/num_i + sum(logZetaSelfSim(num_i+1:end))/num_j;

end