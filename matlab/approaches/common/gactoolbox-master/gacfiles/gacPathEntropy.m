function clusterComp = gacPathEntropy (subIminuszW)
%% Compute structural complexity from the subpart of the weighted adjacency matrix
% Input:
%   - subIminuszW: the subpart of (I - z*P)
% Output:
%	- clusterComp - strucutral complexity of a cluster.
% by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

N = size(subIminuszW,1);
clusterComp = subIminuszW \ ones(N,1);
clusterComp = sum(clusterComp(:)) / (N*N);

end
