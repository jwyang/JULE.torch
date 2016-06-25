function clusterLabels = gacMerging(graphW, initClusters, groupNumber, strDescr, z)
%% Cluster merging for Graph Agglomerative Clustering 
% Implements an agglomerative clustering algorithm based on maiximum graph
%   strcutural affinity of two groups
% Inputs:
%	- graphW: asymmetric weighted adjacency matrix
%   - initClusters: a cell array of clustered vertices
%   - groupNumber: the final number of clusters
%   - strDescr: structural descriptor, 'zeta' or 'path'
%   - z: (I - z*P), default: 0.01
% Outputs:
%   - clusterLabels: 1 x m list whose i-th entry is the group assignment of
%                   the i-th data vector w_i. Groups are indexed
%                   sequentially, starting from 1. 
% by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

%% 
numSample = size(graphW,1);
IminuszW = eye(numSample) - z*graphW;
clear graphW
myInf = 1e10;

%% initialization
VERBOSE = true;
switch lower(strDescr)
    case 'zeta'
        complexity_fun = @gacZetaEntropy;
        conditionalComplexity_fun = @gacZetaCondEntropy;
    case 'path'
        complexity_fun = @gacPathEntropy;
        conditionalComplexity_fun = @gacPathCondEntropy;
    otherwise
        error('GAC: Descriptor type is not supported!');
end

numClusters = length(initClusters);
if numClusters <= groupNumber
    error('GAC: too few initial clusters. Do not need merging!');
end

%% compute the structural complexity of each initial cluster
clusterComp = zeros(numClusters,1);
for i = 1 : numClusters
    clusterComp(i) = complexity_fun(IminuszW(initClusters{i}, initClusters{i}));
end

%% compute initial (negative) affinity table (upper trianglar matrix), very slow
if VERBOSE
    disp('   Computing initial table.' );
end
affinityTab = Inf(numClusters);
for j = 1 : numClusters
    for i = 1 : j-1
        affinityTab(i, j) = - conditionalComplexity_fun(IminuszW, initClusters{i}, initClusters{j}); 
    end
end
affinityTab = bsxfun(@plus, clusterComp, clusterComp') + affinityTab;

if VERBOSE
    disp('   Starting merging process');
end

curGroupNum = numClusters;
while true 
    if mod( curGroupNum, 20 ) == 0 && VERBOSE
        disp(['   Group count: ' num2str(curGroupNum)]);
    end
    % Find two clusters with the best affinity
    [minAff, minIndex1] = min(affinityTab(1:curGroupNum, 1:curGroupNum), [], 1);
    [~, minIndex2] = min(minAff);
    minIndex1 = minIndex1(minIndex2);
    if minIndex2 < minIndex1,  [minIndex1, minIndex2] = swap(minIndex1, minIndex2); end

    % merge the two clusters
    new_cluster = unique([initClusters{minIndex1}; initClusters{minIndex2}]);
    % move the second cluster to be merged to the end of the cluster array
    % note that we only need to copy the end cluster's information to
    % the second cluster's position
    if (minIndex2 ~= curGroupNum)
        initClusters{minIndex2} = initClusters{end};
        clusterComp(minIndex2) = clusterComp(curGroupNum);
        % affinityTab is an upper triangular matrix
        affinityTab(1:minIndex2-1, minIndex2) = affinityTab(1:minIndex2-1, curGroupNum);
        affinityTab(minIndex2, minIndex2+1:curGroupNum-1) = affinityTab(minIndex2+1:curGroupNum-1, curGroupNum);
    end
    
    % update the first cluster and remove the second cluster
    initClusters{minIndex1} = new_cluster;
    initClusters(end) = [];
    clusterComp(minIndex1) = complexity_fun(IminuszW(new_cluster, new_cluster));
    clusterComp(curGroupNum) = myInf;
    affinityTab(:,curGroupNum) = myInf;
    affinityTab(curGroupNum,:) = myInf;
    curGroupNum = curGroupNum - 1;
    if curGroupNum <= groupNumber
        break;
    end

    % update the affinity table for the merged cluster
    for groupIndex1 = 1:minIndex1-1
        affinityTab(groupIndex1, minIndex1) = - conditionalComplexity_fun(IminuszW, initClusters{groupIndex1}, new_cluster);
    end
    for groupIndex1 = minIndex1+1:curGroupNum
        affinityTab(minIndex1, groupIndex1) = - conditionalComplexity_fun(IminuszW, initClusters{groupIndex1}, new_cluster);
    end
    affinityTab(1:minIndex1-1, minIndex1) = clusterComp(1:minIndex1-1) + clusterComp(minIndex1) + affinityTab(1:minIndex1-1, minIndex1);
    affinityTab(minIndex1, minIndex1+1:curGroupNum) = clusterComp(minIndex1+1:curGroupNum)' + clusterComp(minIndex1) + affinityTab(minIndex1, minIndex1+1:curGroupNum);
end

%% generate sample labels
clusterLabels = ones(numSample,1);
for i = 1:length(initClusters)
    clusterLabels(initClusters{i}) = i;
end
if VERBOSE
    disp(['   Final group count: ' num2str(curGroupNum)]);
end

end

function [y, x] = swap (x, y)
end