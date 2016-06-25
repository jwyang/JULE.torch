


%%%%% load dataset and cluster using spectral clustering
clear;

mex dist2aff.cpp ;
mex evrot.cpp  ;
mex scale_dist.cpp ;


%%%%%%%%%%%%%%%% Parameters
neighbor_num = 15;         %% Number of neighbors to consider in local scaling
scale = 0.04;              %% Scale to use for standard spectral clsutering

%figure(1); clf;            %% display parameters
figure;
colors = [1,0,0;0,1,0;0,0,1;1,1,0;1,0,1;0,1,1;0,0,0];

%%%%%%%%%%%%%%% load data
load Data6.mat;
 
%%%%%%%%%%%%%
AUTO_CHOOSE = 1;
CLUSTER_NUM_CHOICES = [2,3,4,5];

%%%%%%%%%%%%%%% cluster all datasets
for j=1:length(XX)
    X = XX{j};
    nGroups = group_num(j);
     
    %% centralize and scale the data
    X = X - repmat(mean(X),size(X,1),1);
    X = X/max(max(abs(X)));

    %%%%%%%%%%%%%%%%% Build affinity matrices
    D = dist2(X,X);              %% Euclidean distance
    A = exp(-D/(scale^2));       %% Standard affinity matrix (single scale)
    [D_LS,A_LS,LS] = scale_dist(D,floor(neighbor_num/2)); %% Locally scaled affinity matrix
    clear D_LS; clear LS;
    
    %% Zero out diagonal
    ZERO_DIAG = ~eye(size(X,1));
    A = A.*ZERO_DIAG;
    A_LS = A_LS.*ZERO_DIAG;

    %%%%%%%%%%%%%%%%% Clustering
    %%%%%%%%%%%%%%% Standard spectral clustering (STD)
    clusts_STD = gcut(A,nGroups);
 

    %%%%%%%%%%%%%%% ZelnikPerona Locally Scaled clustering (LS)
    clusts_LS = gcut(A_LS,nGroups);

    %%%%%%%%%%%%%%% ZelnikPerona Rotation clustering with local scaling (RLS)
    if(AUTO_CHOOSE == 0)
        clusts_RLS = cluster_rotate(A_LS,nGroups);
        rlsBestGroupIndex = 1;
    else
        [clusts_RLS, rlsBestGroupIndex, qualityRLS] = cluster_rotate(A_LS,CLUSTER_NUM_CHOICES,0,1);
        fprintf('column %d\n', j);
        fprintf('RLS qualities: \n');
        qualityRLS
        fprintf('RLS automatically chose best group index as %d (%d clusters)\n', rlsBestGroupIndex, length(clusts_RLS{rlsBestGroupIndex}));
    end

    %%%%%%%%%%%%%%% ZelnikPerona Rotation clustering without local scaling (R)
    if(AUTO_CHOOSE == 0)
        clusts_R = cluster_rotate(A,nGroups);
        rBestGroupIndex = 1;
    else
        [clusts_R, rBestGroupIndex, qualityR] = cluster_rotate(A, CLUSTER_NUM_CHOICES);
        %fprintf('R qualities: \n');
        %qualityR
        %fprintf('R automatically chose best group index as %d (%d clusters)\n', rBestGroupIndex, length(clusts_R{rBestGroupIndex}));
    end
    
        
    
    
    %%%%%%%%%%%% display results
    subplot(length(XX),4,1+(j-1)*4);
    hold on;
    for i=1:length(clusts_STD),
        plot(X(clusts_STD{i},1),X(clusts_STD{i},2),'.','Color',colors(i,:),'MarkerSize',16);
    end
    axis equal;
    title('Standard');
    hold off;
    drawnow;

    subplot(length(XX),4,2+(j-1)*4);
    hold on;
    for i=1:length(clusts_LS),
        plot(X(clusts_LS{i},1),X(clusts_LS{i},2),'.','Color',colors(i,:),'MarkerSize',16);
    end
    axis equal;
    title('Local Scaling');
    hold off;
    drawnow;

    subplot(length(XX),4,3+(j-1)*4);
    hold on;
    for i=1:length(clusts_RLS{rlsBestGroupIndex}),
        plot(X(clusts_RLS{rlsBestGroupIndex}{i},1),X(clusts_RLS{rlsBestGroupIndex}{i},2),'.','Color',colors(i,:),'MarkerSize',16);
    end
    axis equal;
    title('Rotation with Local Scaling');
    hold off;
    drawnow;

    subplot(length(XX),4,4+(j-1)*4);
    hold on;
    for i=1:length(clusts_R{rBestGroupIndex}),
        plot(X(clusts_R{rBestGroupIndex}{i},1),X(clusts_R{rBestGroupIndex}{i},2),'.','Color',colors(i,:),'MarkerSize',16);
    end
    axis equal;
    title('Rotation without Local Scaling');
    hold off;
    drawnow;

end












