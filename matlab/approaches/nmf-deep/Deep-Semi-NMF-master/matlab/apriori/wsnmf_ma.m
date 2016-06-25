 function [  Z, H, dnorm, W, D] = wsnmf_ma( X, ys, k, varargin )
% ssnmf( X, y, k, varargin )

% Process optional arguments
pnames = {'z0' 'h0' 'bUpdateH' 'maxiter' 'TolFun' 'bUpdateZ' 'verbose' 'lambda' 'save' 'correlation'};


dflts  = {0, 0, 1, 300,  1e-6, 1, 1, {}, 0, 0};

[Z, H, bUpdateH, max_iter, tolfun, bUpdateZ, verbose, lambda, doSave, correlation] = ...
        internal.stats.parseArgs(pnames,dflts,varargin{:});

H = LPinitSemiNMF(X, k); 

if ~iscell(Z)
Z = X * pinv(H);
end

nSmp = size(X, 2);
W = zeros(nSmp, nSmp);
cell
for i = 1:length(ys);    
    options = [];
    options.NeighborMode = 'Supervised';
    options.gnd = ys{i};
    options.WeightMode = 'HeatKernel';
    options.t = 1;
    W = W + lambda{i} * constructW(X', options) ;
end

W = W - correlation;

DCol = full(sum(W,2));
D = spdiags(DCol,0,nSmp,nSmp);   
L = D - W;


for i = 1:max_iter;
    try
    if bUpdateZ
        Z = X * pinv(H);
    end
    
    catch
        display('Error inverting Z');
    end
    
    A = Z' * X;
    Ap = (abs(A)+A)*0.5;
    An = (abs(A)-A)*0.5;
    
    B = Z' * Z;
    Bp = (abs(B)+B)*0.5;
    Bn = (abs(B)-B)*0.5;
    
    if bUpdateH
        H = H .* ((Ap + Bn * H + H * W ) ./ max(An + Bp * H +  H * D, eps)).^0.5;
    end
    
    if mod(i, 50) == 0
       display(i); 
    end
      
    if mod(i, 10) == 0 || mod(i+1, 10) == 0 
        
       
        %dnorm = sqrt(sum(s(:).^2)) + lambda * trace(H * L * H');
        dnorm = norm(X - Z * H, 'fro') + trace(H * L * H');


        if mod(i, 100) == 0 && verbose
            display(sprintf('...SS-NMF iteration #%d out of %d, error: %f\n', i, max_iter, dnorm));
        end

        if 0 && exist('dnorm0')
            assert(dnorm <= dnorm0, sprintf('Rec. error increasing! From %f to %f. (%d)', dnorm0, dnorm, k));
        end

        % Check for convergence
        if exist('dnorm0') && 0 && dnorm0-dnorm <= tolfun*max(1,dnorm0)
            if verbose
                display(sprintf('Stopped at %d: dnorm: %f, dnorm0: %f', i, dnorm, dnorm0));
            end
            break;
        end
        if mod(i+1, 10) == 0
            dnorm0 = dnorm;
            continue
        end
    end
end

dnorm = norm(X - Z * H, 'fro');


