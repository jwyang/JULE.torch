function clusterComp = gacZetaEntropy (subIminuszW)
%% Compute structural complexity from the subpart of the weighted adjacency matrix
% Input:
%   - subIminuszW: the subpart of (I - z*P)
% Output:
%	- clusterComp - strucutral complexity of a cluster.
% by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

clusterComp = sum(log(real(diag(inv(subIminuszW))))) / size(subIminuszW,1); % x_c in Thm. 2

end