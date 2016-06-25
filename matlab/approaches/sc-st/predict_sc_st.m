function label_pre = predict_sc_st(feat, nClass)
%PREDICT_SC_ST 此处显示有关此函数的摘要
%   此处显示详细说明
%%%%%%%%%%%%%%%% Parameters
AUTO_CHOOSE = 0;
neighbor_num = 150;         %% Number of neighbors to consider in local scaling
scale = 0.04;              %% Scale to use for standard spectral clsutering

%% centralize and scale the data
% feat = bsxfun(@rdivide, feat, sqrt(sum(feat.^2, 2)));
% X = feat - repmat(mean(feat), size(feat,1), 1);
% X = X / max(max(abs(X)));
X = feat;

%%%%%%%%%%%%%%%%% Build affinity matrices
D = dist2(X,X);              %% Euclidean distance
% scale = mean(D(:));
A = exp(-D/(scale));       %% Standard affinity matrix (single scale)
[D_LS, A_LS, LS] = scale_dist(D,floor(neighbor_num/2)); %% Locally scaled affinity matrix
clear D_LS; clear LS;

%% Zero out diagonal
ZERO_DIAG = ~eye(size(X,1));
A = A.*ZERO_DIAG;
A_LS = A_LS.*ZERO_DIAG;

%%%%%%%%%%%%%%%%% Clustering
%%%%%%%%%%%%%%% Standard spectral clustering (STD)
% clusts_STD = gcut(A, nClass);

%%%%%%%%%%%%%%% ZelnikPerona Locally Scaled clustering (LS)
clusts_LS = gcut(A_LS, nClass);

% %%%%%%%%%%%%%%% ZelnikPerona Rotation clustering with local scaling (RLS)
if(AUTO_CHOOSE == 0)
    clusts_RLS = cluster_rotate(A_LS, nClass);
    rlsBestGroupIndex = 1;
else
    CLUSTER_NUM_CHOICES = nClass;
    [clusts_RLS, rlsBestGroupIndex, qualityRLS] = cluster_rotate(A_LS,CLUSTER_NUM_CHOICES, 0, 1);
    fprintf('column %d\n', j);
    fprintf('RLS qualities: \n');
    qualityRLS
    fprintf('RLS automatically chose best group index as %d (%d clusters)\n', rlsBestGroupIndex, length(clusts_RLS{rlsBestGroupIndex}));
end

%%%%%%%%%%%%%%% ZelnikPerona Rotation clustering without local scaling (R)
% if(AUTO_CHOOSE == 0)
%     clusts_R = cluster_rotate(A, nClass);
%     rBestGroupIndex = 1;
% else
%     [clusts_R, rBestGroupIndex, qualityR] = cluster_rotate(A, CLUSTER_NUM_CHOICES);
%     %fprintf('R qualities: \n');
%     %qualityR
%     %fprintf('R automatically chose best group index as %d (%d clusters)\n', rBestGroupIndex, length(clusts_R{rBestGroupIndex}));
% end

clusts = clusts_LS;

label_pre = zeros(1, size(feat, 1));

for i = 1:length(clusts)
    label_pre(clusts{i}) = i;
end
    
end

