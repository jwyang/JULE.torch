function initialClusters = gacNNMerge(distance_matrix, NNIndex)
% merge each vertex with its nearest neighbor
% by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011
%

%% NN indices
sampleNum = size(distance_matrix,1);
if nargin < 2 || size(NNIndex,1) ~= sampleNum || size(NNIndex,2) < 2
    [~, NNIndex] = sort(distance_matrix, 2);
end

%%
clusterLabels = zeros(sampleNum, 1);
counter = 1;
for i = 1 : sampleNum
    idx = NNIndex(i, 1:2);
    assignedCluster = clusterLabels(idx);
    assignedCluster = unique(assignedCluster(assignedCluster > 0));
    switch length(assignedCluster)
        case 0
            clusterLabels(idx) = counter;
            counter = counter + 1;
        case 1
            clusterLabels(idx) = assignedCluster;
        otherwise
            clusterLabels(idx) = assignedCluster(1);
            for j = 2 : length(assignedCluster)
                clusterLabels(clusterLabels == assignedCluster(j)) = assignedCluster(1);
            end
    end
end
% [graphW, ~] = gacBuildDigraph_c(distance_matrix, 1, 0.95);
% [~, clusterLabels] = graphconncomp(sparse(graphW), 'Directed', true, 'Weak', true);

uniqueLabels = unique(clusterLabels);
% disp(uniqueLabels);
clusterNumber = length(uniqueLabels);
% mappings = zeros(counter, 1);
% mappings(uniqueLabels) = [1:clusterNumber];
% clusterLabels = mappings(clusterLabels);

initialClusters = cell(clusterNumber,1);
for i = 1 : clusterNumber
    initialClusters{i} = find(clusterLabels(:) == uniqueLabels(i));
end

end