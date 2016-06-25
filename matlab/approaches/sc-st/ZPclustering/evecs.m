
function [V,ss,L] = evecs(A,nEvecs)

%% calculate eigenvectors, eigenvalues of the laplaican of A
%%
%%   [V,ss,L] = evecs(A,nEvecs)
%%  
%%  Input:
%%        A = Affinity matrix
%%        nEvecs = number of eigenvectors to compute
%%        
%%  Output:       
%%        V = eigenvectors
%%        ss = eigenvalues
%%        L = Laplacian
%%
%%
%%  Code by Lihi Zelnik-Manor (2005)
%%
%%



%%%%%%%% Compute the Laplacian
tic;
npix = size(A,1);
useSparse = issparse(A);
dd = 1./(sum(A)+eps);
dd = sqrt(dd);
if(useSparse)
    DD = sparse(1:npix,1:npix,dd);
else
    DD = diag(dd);
end
L = DD*A*DD;
ttt = toc;
% disp(['Laplacian computation took ' num2str(ttt) ' seconds']);


%%%%%%% Compute eigenvectors
tic;
if (useSparse)
    opts.issym = 1;
    opts.isreal = 1;
    opts.disp = 0;
    [V,ss] = eigs(L,nEvecs,1,opts);
%     [VV,ss]=svds(L,nClusts,1,opts);
else
    [V,ss] = svd(L);
    V = V(:,1:nEvecs);    
end
ss = diag(ss);
ss = ss(1:nEvecs);
ttt = toc;
% disp(['eigenvectors computation took ' num2str(ttt) ' seconds']);




