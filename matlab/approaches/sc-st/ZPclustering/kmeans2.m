function [clusts,bestD]=kmeans2(V,nClusts)


%%%%%%%%%%%%%%%%%% Kmeans
%%%%%% Try 10 runs of k-means and save the one with minimal distortion
bestC = {};           %% a variable to keep the best clustering so far
bestD = 1/eps;        %% a variable to remember the best distortion so far
for nRuns=1:10
    %%%%%% Initialize centers 
    %% First center is set to one entry picked randomly 
    [dd,pp] = max(rand(size(V,1),1)); 
    mu(1,:) = V(pp,:);
    %% The other centers are selected to be farthest from previous centers
    for i=2:nClusts
        ip = V*mu';
        minip = max(abs(ip'));
        [yy,ii] = min(minip);
        mu(i,:) = V(ii,:);
    end
    %%%%%%%%%%% and now run K means
    for tt=1:10   %% 10 iterations for kmeans
        distM = dist2(V,mu);        %% initialize distance between points and centers
        [yy,ii] = min(distM');      %% assign points to nearest center

        distort = 0;
        distort_across = 0;
        clear clusts;
        for nn=1:nClusts
            I = find(ii==nn);       %% indices of points in cluster nn
            J = find(ii~=nn);       %% indices of points not in cluster nn
            clusts{nn} = I;         %% save into clusts cell array
            if (length(I)>0)
                mu(nn,:) = mean(V(I,:));               %% update mean
                %% Compute within class distortion
                muB = repmat(mu(nn,:),length(I),1);
                distort = distort+sum(sum((V(I,:)-muB).^2));
                %% Compute across class distortion
                muB = repmat(mu(nn,:),length(J),1);
                distort_across = distort_across + sum(sum((V(J,:)-muB).^2));
            end
        end
        %% Set distortion as the ratio between the within
        %% class scatter and the across class scatter
        distort = distort/(distort_across+eps);
        if (distort<bestD)   %% save result if better than the best so far
            bestD=distort;
            bestC=clusts;
        end
    end
end

%% Finally, delete empty clusters
pp=1;
for nn=1:nClusts
    if (length(bestC{nn})>0)
        clusts{pp} = bestC{nn};
        pp = pp+1;
    end
end
distortion  = bestD;



