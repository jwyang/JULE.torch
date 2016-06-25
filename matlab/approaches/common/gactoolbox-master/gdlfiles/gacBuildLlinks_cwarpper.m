function outputClusters = gacBuildLlinks_cwarpper(distance_matrix, p, NNIndex)
%% Build l-links by NN propagation (ALL NNs PROPAGATION);
% Two clusters will be merged if their intersection is not empty.
% Inputs:
%        distance_matrix - dis-similarity matrix, i --> j
%        K - number of NNs, not including the pivot points
%        a - parameter of covariance
%
% Output:
%        outputClusters - of format n*1 cell, each element is indices of initially merged clusters 
% by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011
%
% Please cite the following paper, if you find the code is helpful
%
% W. Zhang, X. Wang, D. Zhao, and X. Tang. 
% Graph Degree Linkage: Agglomerative Clustering on a Directed Graph.
% in Proceedings of European Conference on Computer Vision (ECCV), 2012.

%% NN indices
if nargin < 4 || size(NNIndex,1) ~= size(distance_matrix,1) || size(NNIndex,2) < (p+1)
    [~,NNIndex] = gacMink(distance_matrix,p+1,2);
end

%% find l-links
if p == 1
    outputClusters = gacOnelink_c (double(NNIndex));
else
    outputClusters = gacLlinks_c (double(distance_matrix), double(NNIndex), double(p));
end
disp(['   Initial group count: ' num2str( length(outputClusters) ) ]);

end