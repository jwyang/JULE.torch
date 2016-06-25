
function [clusts,best_group_index,Quality,Vr] = cluster_rotate(A,group_num,fig,method)

%% cluster by rotating eigenvectors to align with the canonical coordinate
%% system
%%
%%   [clusts,best_group_index,Quality,Vr] = cluster_rotate(A,group_num,method,fig)
%%  
%%  Input:
%%        A = Affinity matrix
%%        group_num - an array of group numbers to test
%%                    it is assumed to be a continuous set
%%        fig - Figure to display progress. set to 0 if no display is
%%              desired
%%        method - 1   gradient descent 
%%                 2   approximate gradient descent
%%        
%%  Output:
%%        clusts - a cell array of the results for each group number
%%        best_group_index - the group number index with best alignment quality
%%        Quality = the final quality for all tested group numbers
%%        Vr = the rotated eigenvectors
%%
%%
%%  Code by Lihi Zelnik-Manor (2005)
%%
%%


if( nargin < 2 )
    group_num = [2:6];
end
if( nargin < 3 )
    fig = 0;
end
if( nargin < 4 )
    method = 1;  %% method to calculate cost gradient. 1 means true derivative
                 %% change to any other value to estimate fradient numerically
end
group_num = sort(group_num);
group_num = setdiff(group_num,1);

%%% obtain eigenvectors of laplacian of affinity matrix
tic; 
nClusts = max(group_num);
[V,evals] = evecs(A,nClusts); 
ttt = toc;
disp(['evecs took ' num2str(ttt) ' seconds']);

%%%%%% Rotate eigenvectors
clear clusts;
Vcurr = V(:,1:group_num(1));
for g=1:length(group_num),
    %%% make it incremental (used already aligned vectors)
    if( g > 1 )
        Vcurr = [Vr{g-1},V(:,group_num(g))];
    end
    [clusts{g},Quality(g),Vr{g}] = evrot(Vcurr,method);
end
i = find(max(Quality)-Quality <= 0.001);
best_group_index = i(numel(i));







