function clusterLabels = gdlMergingKNN_c(graphW, initClusters, groupNumber)
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
myBoundInf = 1e8;
Kc = 10;

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
[affinityTab, AsymAffTab] = gdlInitAffinityTable_knn_c (double(graphW), initClusters, Kc);
affinityTab = - affinityTab;
affinityTab(1:numClusters+1:end) = myInf;
% [affinityTab, AsymAffTab] = gdlInitAffinityTable_c (double(graphW), initClusters);
% affinityTab = tril(myInf*ones(numClusters)) - affinityTab;
% AsymAffTab(1,2) - 1->2->1, AsymAffTab(2,1) - 2->1->2
[~, KcCluster] = gacMink(affinityTab, Kc);

if VERBOSE
    disp('   Starting merging process');
end

curGroupNum = numClusters;
while true
    usingKcCluster = curGroupNum > 1.2*Kc;
    if mod( curGroupNum, 50 ) == 0 && VERBOSE
        disp(['   Group count: ' num2str(curGroupNum)]);
    end
    % Find two clusters with the best affinity
%     [~, minIndex] = min(reshape(affinityTab(1:curGroupNum,1:curGroupNum), 1, []));
%     [minIndex1, minIndex2] = ind2sub([curGroupNum,curGroupNum], minIndex);
%     if (minIndex1 > minIndex2),  [minIndex1, minIndex2] = swap(minIndex1, minIndex2);  end
    [~, minIndex1, minIndex2] = gacPartialMin_knn_c(affinityTab, curGroupNum, KcCluster);  % the indices are sorted
    cluster1 = initClusters{minIndex1};
    cluster2 = initClusters{minIndex2};

    % merge the two clusters
    new_cluster = unique([cluster1; cluster2]);
    % find candidates to be updated
    if usingKcCluster
        KcCluster(KcCluster == minIndex2) = minIndex1;
        candidates = (any(KcCluster == minIndex1));
        candidates([KcCluster(:,minIndex1); KcCluster(:,minIndex2)]) = true;
        candidates([minIndex1, minIndex2]) = false;
        candidates = find(candidates);
    end
    % move the second cluster to be merged to the end of the cluster array
    % note that we only need to copy the end cluster's information to
    % the second cluster's position
    if (minIndex2 ~= curGroupNum)
        initClusters{minIndex2} = initClusters{end};
        % affinityTab is an upper triangular matrix
        affinityTab(1:curGroupNum-1, minIndex2) = affinityTab(1:curGroupNum-1, curGroupNum);
        affinityTab(minIndex2, 1:curGroupNum-1) = affinityTab(curGroupNum, 1:curGroupNum-1);
        if usingKcCluster
            KcCluster(:,minIndex2) = KcCluster(:,end);
            KcCluster(KcCluster == curGroupNum) = minIndex2;
            candidates(candidates == curGroupNum) = minIndex2;
        end
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
    if usingKcCluster
        KcCluster(:,end) = [];
    end
    curGroupNum = curGroupNum - 1;
    if curGroupNum <= groupNumber
        break;
    end

    % update the affinity table for the merged cluster
    if usingKcCluster
        affinityTab(1:curGroupNum,minIndex1) = myInf;
        for candIdx = 1:length(candidates)
            groupIndex1 = candidates(candIdx);
            if AsymAffTab(minIndex1, groupIndex1) > -myBoundInf && AsymAffTab(groupIndex1, minIndex1) > -myBoundInf
                AsymAffTab(minIndex1, groupIndex1) = gdlDirectedAffinity_c (graphW, initClusters, minIndex1, groupIndex1);
            else
                [~, AsymAffTab(groupIndex1, minIndex1), AsymAffTab(minIndex1, groupIndex1)] = gdlAffinity_c (graphW, initClusters{groupIndex1}, new_cluster);
            end
        end
        affinityTab(candidates, minIndex1) = - AsymAffTab(minIndex1, candidates)' - AsymAffTab(candidates, minIndex1); 
    else
        affinityTab(minIndex1,minIndex1) = myInf;
        for groupIndex1 = 1:curGroupNum
            if groupIndex1 == minIndex1, continue; end
            if AsymAffTab(minIndex1, groupIndex1) > -myBoundInf && AsymAffTab(groupIndex1, minIndex1) > -myBoundInf
                AsymAffTab(minIndex1, groupIndex1) = gdlDirectedAffinity_c (graphW, initClusters, minIndex1, groupIndex1);
            else
                [~, AsymAffTab(groupIndex1, minIndex1), AsymAffTab(minIndex1, groupIndex1)] = gdlAffinity_c (graphW, initClusters{groupIndex1}, new_cluster);
            end
        end
        affinityTab(1:curGroupNum, minIndex1) = - AsymAffTab(minIndex1, 1:curGroupNum)' - AsymAffTab(1:curGroupNum, minIndex1);
    end
    affinityTab(minIndex1,1:curGroupNum) = affinityTab(1:curGroupNum,minIndex1)';
    if usingKcCluster
        [~, KcCluster(:,minIndex1)] = gacMink(affinityTab(1:curGroupNum,minIndex1), Kc);
    end
%     disp([minIndex1])
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