function clusterLabels = gdlMerging_c(graphW, initClusters, groupNumber)
%% Cluster merging for Graph Degree Linkage 
% Implements an agglomerative clustering algorithm based on maiximum graph
%   strcutural affinity of two groups
% Inputs:
%	- graphW: asymmetric weighted adjacency matrix
%   - initClusters: a cell array of clustered vertices
%   - groupNumber: the final number of clusters
% Outputs:
%   - clusterLabels: 1 x m list whose i-th entry is the group assignment of
%                   the i-th data vector w_i. Groups are indexed
%                   sequentially, starting from 1. 
% by Wei Zhang (wzhang009 at gmail.com), Nov., 7, 2011
%
% Please cite the following paper, if you find the code is helpful
%
% W. Zhang, X. Wang, D. Zhao, and X. Tang. 
% Graph Degree Linkage: Agglomerative Clustering on a Directed Graph.
% in Proceedings of European Conference on Computer Vision (ECCV), 2012.

%% 
numSample = size(graphW,1);
myInf = 1e10;

%% initialization
VERBOSE = false;

numClusters = length(initClusters);
if numClusters <= groupNumber
    error('GAC: too few initial clusters. Do not need merging!');
end

%% compute initial (negative) affinity table (upper trianglar matrix), very slow
if VERBOSE
    disp('   Computing initial table.' );
end
[affinityTab, AsymAffTab] = gdlInitAffinityTable_c (double(graphW), initClusters);
affinityTab = tril(myInf*ones(numClusters)) - affinityTab;
% AsymAffTab(1,2) - 1->2->1, AsymAffTab(2,1) - 2->1->2

if VERBOSE
    disp('   Starting merging process');
end

curGroupNum = numClusters;
while true 
    if mod( curGroupNum, 50 ) == 0 && VERBOSE
        disp(['   Group count: ' num2str(curGroupNum)]);
    end
    % Find two clusters with the best affinity
    [~, minIndex1, minIndex2] = gacPartialMin_triu_c(affinityTab, curGroupNum);  % the indices are sorted
    cluster1 = initClusters{minIndex1};
    cluster2 = initClusters{minIndex2};

    % merge the two clusters
    new_cluster = unique([cluster1; cluster2]);
    % move the second cluster to be merged to the end of the cluster array
    % note that we only need to copy the end cluster's information to
    % the second cluster's position
    if (minIndex2 ~= curGroupNum)
        initClusters{minIndex2} = initClusters{end};
        % affinityTab is an upper triangular matrix
        affinityTab(1:minIndex2-1, minIndex2) = affinityTab(1:minIndex2-1, curGroupNum);
        affinityTab(minIndex2, minIndex2+1:curGroupNum-1) = affinityTab(minIndex2+1:curGroupNum-1, curGroupNum);
    end
%     AsymAffTab(minIndex1, 1:curGroupNum) = AsymAffTab(minIndex1, 1:curGroupNum) + AsymAffTab(minIndex2, 1:curGroupNum);
    AsymAffTab(1:curGroupNum, minIndex1) = AsymAffTab(1:curGroupNum, minIndex1) + AsymAffTab(1:curGroupNum, minIndex2);
    AsymAffTab(1:curGroupNum, minIndex2) = AsymAffTab(1:curGroupNum, curGroupNum);
    AsymAffTab(minIndex2, 1:curGroupNum) = AsymAffTab(curGroupNum, 1:curGroupNum);

    % update the first cluster and remove the second cluster
    initClusters{minIndex1} = new_cluster;
    initClusters(end) = [];
    affinityTab(:,curGroupNum) = myInf;
    affinityTab(curGroupNum,:) = myInf;
    curGroupNum = curGroupNum - 1;
    if curGroupNum <= groupNumber
        break;
    end

    % update the affinity table for the merged cluster
% % % The commented matlab code is what we do in the following mex function
% %     degProd = sum(graphW(new_cluster, :), 1) .* sum(graphW(:, new_cluster), 2)';
% %     for groupIndex1 = 1:curGroupNum
% %         if groupIndex1 == minIndex1, continue; end
% %         AsymAffTab(minIndex1, groupIndex1) = sum(degProd(initClusters{groupIndex1}));
% %     end
% %     AsymAffTab(minIndex1, 1:curGroupNum) = AsymAffTab(minIndex1, 1:curGroupNum) / (length(new_cluster)*length(new_cluster));
%     for groupIndex1 = 1:curGroupNum
%         if groupIndex1 == minIndex1, continue; end
%         AsymAffTab(minIndex1, groupIndex1) = gdlDirectedAffinity_c (graphW, initClusters, minIndex1, groupIndex1);
%     end
    AsymAffTab(minIndex1, 1:curGroupNum) = gdlDirectedAffinity_batch_c (graphW, initClusters, minIndex1);
    affinityTab(1:minIndex1-1, minIndex1) = - AsymAffTab(minIndex1, 1:minIndex1-1)' - AsymAffTab(1:minIndex1-1, minIndex1);
    affinityTab(minIndex1, minIndex1+1:curGroupNum) = - AsymAffTab(minIndex1, minIndex1+1:curGroupNum) - AsymAffTab(minIndex1+1:curGroupNum, minIndex1)';
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