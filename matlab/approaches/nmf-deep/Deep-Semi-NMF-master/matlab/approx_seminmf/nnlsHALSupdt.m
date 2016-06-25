function V = nnlsHALSupdt(M,U,V,maxiter)
% V = nnlsHALSupdt(M,U,V,maxiter)
%
% Computes an approximate solution of the following nonnegative least 
% squares problem (NNLS)  
%
%           min_{V >= 0} ||M-UV||_F^2 
% 
% with an exact block-coordinate descent scheme. 
%
% See N. Gillis and F. Glineur, Accelerated Multiplicative Updates and 
% Hierarchical ALS Algorithms for Nonnegative Matrix Factorization, 
% Neural Computation 24 (4): 1085-1105, 2012.
% 
%
% ****** Input ******
%   M  : m-by-n matrix 
%   U  : m-by-r matrix
%   V  : r-by-n initialization matrix 
%        default: one non-zero entry per column corresponding to the 
%        clostest column of U of the corresponding column of M 
%   maxiter: upper bound on the number of iterations (default=500).
%
%   *Remark. M, U and V are not required to be nonnegative. 
%
% ****** Output ******
%   V  : an r-by-n nonnegative matrix \approx argmin_{V >= 0} ||M-UV||_F^2

if nargin <= 3
    maxiter = 500;
end
[m,n] = size(M); 
[m,r] = size(U); 
if nargin <= 2 || isempty(V) 
    V = zeros(r,n); 
    for i = 1 : n
        % Distance between ith column of M and columns of U
        disti = sum( (U - repmat(M(:,i),1,r)).^2 ); 
        [a,b] = min(disti); 
        V(b,i) = 1; 
    end
end

UtU = U'*U; 
UtM = U'*M; 
delta = 1e-6; % Stopping condition depending on evolution of the iterate V: 
              % Stop if ||V^{k}-V^{k+1}||_F <= delta * ||V^{0}-V^{1}||_F 
              % where V^{k} is the kth iterate. 
eps0 = 0; cnt = 1; eps = 1; 
while eps >= (delta)^2*eps0 && cnt <= maxiter %Maximum number of iterations
    nodelta = 0; if cnt == 1, eit3 = cputime; end
        for k = 1 : r
            deltaV = max((UtM(k,:)-UtU(k,:)*V)/UtU(k,k),-V(k,:));
            V(k,:) = V(k,:) + deltaV;
            nodelta = nodelta + deltaV*deltaV'; % used to compute norm(V0-V,'fro')^2;
            if V(k,:) == 0, V(k,:) = 1e-16*max(V(:)); end % safety procedure
        end
    if cnt == 1
        eps0 = nodelta; 
    end
    eps = nodelta; 
    cnt = cnt + 1; 
end

end % of function nnlsHALSupdt