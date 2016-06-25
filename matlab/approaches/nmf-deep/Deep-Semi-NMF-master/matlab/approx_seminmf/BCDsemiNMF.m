function [U,V,e] = BCDsemiNMF(M,V,maxiter)
% [U,V,e] = BCDsemiNMF(M,V,maxiter)
% 
% Block coordiante descent for Semi-NMF
%
% It guarantees the objective function to decrease; see Algorithm 1 in 
% N. Gillis, A. Kumar, Exact and Heuristic Algorithms for Semi-Nonnegative 
% Matrix Factorization, arXiv, 2014
% 
%
% ****** Input ******
%   M      : m-by-n matrix 
%   V      : an r-by-n initialization V 
%   maxiter: a maximum number of iterations
%
% ****** Output ******
%   (U,V)  : a semi-NMF of M \approx UV, V >= 0
%    e     : vector of the evolution of the error at each iteration k

if nargin <= 2
    maxiter = 100;
end
i = 1; 
nM2 = norm(M,'fro')^2; 
while i <= maxiter
    U = M/V; % Optimal solution
    V = nnlsHALSupdt(M,U,V,1); % BCD on the rows of V
    if nargout >= 3
        % e(i) = norm(M-U*V,'fro')... 
        % Better for sparse matrices: 
        e(i) = nM2 - 2*sum(sum( U.*(M*V'))) + sum(sum( (U'*U).*(V*V') ) ); 
        e(i) = sqrt(max(0, e(i))); 
    end
    i = i + 1; 
end