function [ Z, H, dnorm] = nmf ( X, k, varargin )
% Matrix sizes
% X: m x n
% Z: m x num_of_components
% H: num_of_components x num_of_components

% Process optional arguments
pnames = {'z0' 'h0' 'bUpdateH' 'maxiter' 'nonlinearity_function', 'TolFun'};

% Do SVD initialisation of the init components

if 1
    [z0, h0] = NNDSVD(abs(X), k, 0);
else
    z0 = rand(size(X, 1), k);
    h0 = rand(k, size(X,2));
end

dflts  = {z0, h0, 1, 300, @(x) x, 1e-5};

[z0, h0, bUpdateH, max_iter, g, tolfun] = ...
        internal.stats.parseArgs(pnames,dflts,varargin{:});


gX = g(X);

Z = z0;
H = h0;

for i = 1:max_iter
    if bUpdateH
        numer = Z' * gX;
        H = H .* (numer ./ (((Z' * Z) * H) + eps(numer)));
    end
   
    numer = gX * H';
    Z = Z .* (numer ./  ((Z * (H * H')) + eps(numer)));    
    
    if mod(i, 10) == 0 || mod(i+1, 10) == 0 
        s = gX - Z * H;
        dnorm = sqrt(sum(s(:).^2));
        % dnorm = norm(gX - Z * H, 'fro');
        
        if mod(i+1, 10) == 0
            dnorm0 = dnorm;
            continue
        end

        if mod(i, 100) == 0
            display(sprintf('...NMF iteration #%d out of %d, error: %f\n', i, max_iter, dnorm));
        end

        if exist('dnorm0')
            assert(dnorm <= dnorm0, sprintf('Rec. error increasing! From %f to %f. (%d)', dnorm0, dnorm, k));
        end

        % Check for convergence
        if exist('dnorm0') && dnorm0-dnorm <= tolfun*max(1,dnorm0)
            display(sprintf('Stopped at %d: dnorm: %f, dnorm0: %f', i, dnorm, dnorm0));
            break;
        end
     
    end
end


