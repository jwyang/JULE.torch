 function [  Z, H, dnorm, W, D] = dnmf( X, y, k, varargin )
% ssnmf( X, y, k, varargin )

% Process optional arguments
pnames = {'z0' 'h0' 'bUpdateH' 'maxiter' 'TolFun' 'bUpdateZ' 'verbose' 'lambda' 'save'};


dflts  = {0, 0, 1, 300,  1e-6, 1, 1, 10, 1};

[Z, H, bUpdateH, max_iter, tolfun, bUpdateZ, verbose, lambda, doSave] = ...
        internal.stats.parseArgs(pnames,dflts,varargin{:});


[Z, H] = NNDSVD(abs(X), k, 0);

nSmp = size(X, 2);

W = zeros(nSmp, nSmp);

for i=1:length(y);
    W(i, 1:length(y)) =  (y == y(i)) / sum(y == y(i));
end
    

W = lambda * W;
D = lambda * eye(nSmp, nSmp);
L = D - W;

for i = 1:max_iter;
    if bUpdateZ
         numer = X * H' ;
         Z = Z .* (numer ./  ((Z * (H * H') ) + eps(numer)));    
    end

    if bUpdateH
        numer = Z' * X +  H * W;
        H = H .* (numer ./ (((Z' * Z) * H) +  H * D + eps(numer)));    
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

H = H(:, length(y)+1:end);
