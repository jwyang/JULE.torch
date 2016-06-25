
function [mask,clusts,Quality,D,W] = segment_image(IM,R,nGroups,method1,method2,varargin);

%% intensity based image segmentation
%% [mask,clusts,Quality,D,W] = seg_image(IM,R,nGroups,method1,method2,varargin)
%%
%% Input:
%%   IM:           image to segment
%%   R:            neighborhood of connectivity
%%   nGroups:      number of segments
%%   method1:      'SS': use a Single given Scale to switch from
%%                       distances to affinities
%%                 'LS': use automatic Local Scaling (default)
%%   method2:      'KM': use kemans to obtain final clustering
%%                 'RT1': Rotate eigenvectors to align with canonical
%%                        coordinate-frame to obtain final clustering
%%                        Descend through real derivative
%%                 'RT2': Descend through numerical derivative
%%
%% Output:
%%   mask = an integer mask image where each pixel is "colored" according
%%          to its cluster assignment
%%   clusts = a cell array of cluster assignments
%%   Quality = a measure of the final clustering quality
%%   D = the distance matrix
%%   W = the affinity matrix
%%
%%  Lihi Zelnik-Manor, March 2005, Caltech
%%


%%% initialize parameters
if( nargin < 5 )
    method2 = 'RT1';
end
if( nargin < 4 )
    method1 = 'LS';
end
if( nargin < 3 )
    nGroups = 2;
end
if( nargin < 2 )
    R = 5;
end
if( strcmp(method1,'SS') )
    if( nargin < 6 )
        SS = 0.1;
    else
        SS = varargin{1};
    end
end

Quality = 0;
D = [];
W = [];
[rows,cols,colors]=size(IM);

%%% build distance matrix
tic; [D,ind_non_zero,rows_nonz,cols_nonz] = imdist(IM,R); ttt=toc;
disp(['Building affinity matrix took ' num2str(ttt) ' second']);

%%% switch to affinities
if( strcmp(method1,'SS') )
%     tic; W = D;  W(ind_non_zero)=exp(-D(ind_non_zero)/SS^2); ttt = toc;
    tic; W = dist2aff(D,SS); ttt = toc;
    disp(['Exp took ' num2str(ttt) ' seconds']);
elseif( strcmp(method1,'LS') )
    tic; [DN,W, N] = scale_dist(D,7); ttt=toc;
    disp(['Local scaling took ' num2str(ttt) ' seconds']);
else
    error(['WARNING: Unknown method1 ' method1 ' in segment_image']);
end
    

 
%%% obtain final clustering
mask = zeros(rows,cols);
if( strcmp(method2,'KM') )  %% kmeans
    %%% obtain eigenvectors of laplacian of affinity matrix
    tic; [V,evals] = evecs(W,max(nGroups)); ttt = toc;
    disp(['evecs took ' num2str(ttt) ' seconds']);

    tic;
    %% normalize rows of V
    for i=1:size(V,1); V(i,:)=V(i,:)/(norm(V(i,:))+eps);  end
    %% call kmeans
    [clusts,distort] = kmeans2(V,max(nGroups));
    ttt = toc;
    disp(['kmeans took ' num2str(ttt) ' seconds']);  
    for i=1:length(clusts),
        mask(clusts{i}) = i;
    end
    Quality = -distort;
elseif( strcmp(method2,'RT1') ) %% rotation
    tic;
    [clusts,best_group_index,Quality] = cluster_rotate(W,nGroups,0,1);
    ttt = toc;
    disp(['rotation took ' num2str(ttt) ' seconds']);  
    for i=1:length(clusts{best_group_index}),
        mask(clusts{best_group_index}{i}) = i;
    end
elseif( strcmp(method2,'RT2') ) %% rotation
    tic;
    [clusts,best_group_index,Quality] = cluster_rotate(W,nGroups,0,2);
    ttt = toc;
    disp(['approx rotation took ' num2str(ttt) ' seconds']);  
    for i=1:length(clusts{best_group_index}),
        mask(clusts{best_group_index}{i}) = i;
    end
else
    error(['WARNING: Unknown method2 ' method2 ' in segment_image']);
end
    
  
     
















