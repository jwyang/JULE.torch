function [ Z, H, dnorm, R ] = ssdeepnmf ( X, layers, varargin )

% Process optional arguments
pnames = { ...
    'z0' 'h0' 'bUpdateH' 'bUpdateLastH' 'maxiter' 'g', 'g_inv', ...
    'g_inv_diff', 'TolFun', 'save', 'nonlinearity' ...
};

num_of_layers = numel(layers);

% Normalise X to unit norm

X = bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));


Z = cell(1, num_of_layers);
H = cell(1, num_of_layers);

%dflts  = {0, 0, 1, 1, 500, @(x) sqrt(x), @(x) x.*x, @(x) 2 .* x, 1e-5};
dflts  = {0, 0, 1, 1, 300, @(x) x, @(x) x, @(x) x, 1e-5, 1, 'tanh'};

[z0, h0, bUpdateH, bUpdateLastH, maxiter, g, g_inv, g_inv_diff, tolfun, doSave, nonlinearity] = ...
        internal.stats.parseArgs(pnames,dflts,varargin{:});


if strcmp(nonlinearity, 'tanh') == 1
    g = @(x) 3 .* atanh(x./1.7159 ) ./2;
    g_inv = @(x) 1.7159 * tanh( 2/3 * x);
    g_inv_diff = @(x)  1.7159 * 2 / 3 .* (sech((2 .* x) ./ 3) .^ 2);
elseif strcmp(nonlinearity, 'square') == 1
    g = @(x) x .^ 0.5;
    g_inv = @(x) x .* x;
    g_inv_diff = @(x) 2 * x;
elseif strcmp(nonlinearity, 'sigmoid') == 1
    sigmoid = @(x) (1./(1+exp(-x)));
    g_inv = sigmoid;
    g_inv_diff = @(x) sigmoid(x) .* (1 - sigmoid(x));
    g = @(x) log(x ./ (1 - x));
elseif strcmp(nonlinearity, 'softplus') == 1
    g_inv = @(x) log(1 + exp(x));
    g_inv_diff = @(x) exp(x) ./ (1 + exp(x));
    g = @(x) log(exp(x) - 1);
else
    
    assert(strcmp(nonlinearity, 'manual') == 1)
            
end

%g = @(x) x .^ (1/3);
%g_inv = @(x) x .* x .* x;
%g_inv_diff = @(x) 3 .* x .* x;

% g = @(x) atanh(x);
% g_inv = @(x) tanh(x);
% g_inv_diff = @(x) sech(x).^2;

R = cell(numel(layers) - 1, 1);

if ~iscell(z0) && ~iscell(h0)
    for i_layer = 1:length(layers)
        if i_layer == 1
            R{i_layer} = 1; %max(max(X(:)), 1);
            % R{i_layer} = NormaliseFactor(X);
            V = X ./ R{i_layer};
        else 
            R{i_layer} = max(max(max(H{i_layer-1})), 1);
            % R{i_layer} = NormaliseFactor(H{i_layer-1});
            V = g(H{i_layer-1} ./ R{i_layer});
        end
        
        
        display(sprintf('Initialising Layer #%d...', i_layer));
    
        % For the later layers we use nonlinearities as we go from
        % g(H_{k-1}) to Z*H_k
        [Z{i_layer}, H{i_layer}, dnorm] = ...
             seminmf(V, ...
                 layers(i_layer), ...
                 'maxiter', maxiter, ...
                 'bUpdateH', true, 'save', doSave); 
             
%         Z{i_layer} = gpuArray(Z{i_layer});
%         H{i_layer} = gpuArray(H{i_layer});
    end

else
    Z=z0;
    H=h0;
    
    display('Skipping initialization, using provided init matrices...');
end

dnorm0 = norm(X - deep_recon(Z, H, R, g_inv), 'fro');
dnorm = dnorm0;

display(sprintf('#%d error: %f', 0, dnorm0));

for i = 1:numel(layers)
    display(sprintf('R[%d] = %f', i, R{i}));
end

%% Error Propagation
display('Finetuning...');

for iter = 1:maxiter  
    for i = numel(layers):-1:1
        if i == 2    
            KSI = Z{1}' * X;
            PSI = Z{1}' * Z{1} * (R{1} * R{2});
        end
        
        if  bUpdateH && (i < numel(layers) || (i == numel(layers) && bUpdateLastH))  
            
            if i == 1
                % A = Z{1}' * X;
                % B = Z{1}' * ((Z{1} * H{1}));
                
                % [H, ~] = gd_H(X, Z, H, R, B - A, i, g_inv, dnorm);       
                
                H{1} = R{2} * g_inv(Z{2} * H{2});
                % H{i}(H{i} <= 0) = eps;
            else 
                c = R{2} * g_inv_diff(Z{2} * H{2});
                
                A = KSI;
                B = PSI  * g_inv(Z{2} * H{2});
                
                C = Z{2}' * ((B - A) .* c);
                [H, dnorm] = gd_H(X, Z, H, R, C, i, g_inv, dnorm);          
            end
        end
        
        
        
        % dnorm = norm(X - deep_recon(Z, H, R, g_inv), 'fro');
        
        % fprintf(1, 'after H(%d)...#%d error: %f\n', i, iter, dnorm);
        
        assert(~any(any(isnan(H{i}))));
        assert(isreal(H{i}));
        
        if i == 1
          A = X;
          B = Z{1} * H{1};
          
          C = (B - A) * H{1}';
        else
          c = R{2} * g_inv_diff(Z{2} * H{2});
          
          A = KSI;
          B = PSI * g_inv(Z{2} * H{2});
          
          C = ((B - A) .* c) * H{2}';
        end
        
        [Z, dnorm] = gd_Z(X, Z, H, R, C, i, g_inv, dnorm);       
        
        % dnorm = norm(X - deep_recon(Z, H, R, g_inv), 'fro');
        
        % display(sprintf('after Z(%d)...#%d error: %f', i, iter, norm(X - deep_recon(Z, H, R, g_inv), 'fro')));
        
        assert(isreal(Z{i}));
        assert(~any(any(isnan(Z{i}))));
    end
    
    %assert(i == numel(layers));
    
    % dnorm = norm(X - deep_recon(Z, H, R, g_inv), 'fro');
    
    display(sprintf('#%d error: %f', iter, dnorm));
    
%     assert(dnorm <= dnorm0 + 1, ...
%         sprintf('Rec. error increasing! From %f to %f. (%d)', ...
%         dnorm0, dnorm, iter) ...
%     );
    
    if 1 && dnorm0-dnorm <= tolfun*max(1,dnorm0)
        display( ...
            sprintf('Stopped at %d: dnorm: %f, dnorm0: %f', ...
                iter, dnorm, dnorm0 ...
            ) ...
        );
        break;
    end
    
    dnorm0 = dnorm;
end



end


function [Z, dnorm1] = gd_Z(X, Z, H, R, c, i, g_inv, dnorm)
    eta = 0.005;
    oldZ = Z{i};
    iter = 0;
    
    while(1)
        iter = iter + 1;
        eta = eta / 2;
        Z{i} = oldZ - eta .* c;
        
        dnorm1 = norm(X - deep_recon(Z, H, R, g_inv), 'fro');
       
        if  eta < 0.00001
           Z{i} = oldZ; 
           dnorm1 = dnorm;
           break;
        end

        if dnorm1 <= dnorm + eps
            fprintf(1, 'Z(%d) eta: %f iter: %d dnorm: %f\n', i, eta, iter, dnorm1);
            break;
        end                              
        %

    end
    
end


function [H, dnorm1] = gd_H(X, Z, H, R, c, i, g_inv, dnorm)
    eta = .3;
    oldH = H{i};
    iter = 0;
    
    if i == 1
        dnorm = norm(X - R{1} * Z{1} * H{1}, 'fro');
    end
        
    while(1)
        iter = iter + 1;
        eta = eta / 2;
        H{i} = oldH - eta .* c;
        H{i}(H{i} <= 0) = eps;
        
        if i == 1
           dnorm1 =  norm(X - R{1} * Z{1} * H{1}, 'fro');
        else
           dnorm1 = norm(X - deep_recon(Z, H, R, g_inv), 'fro');
        end
        
%         fprintf(1, 'H{%d}: eta: %f dnorm: %f\n', i, eta, dnorm1);

        if  eta < 0.00001
           H{i} = oldH; 
           dnorm1 = dnorm;
           break;
        end
        
        %

        if dnorm1 <= dnorm
            fprintf(1, 'H(%d) eta: %f iter: %d dnorm: %f\n', i, eta, iter, dnorm1);
            break;
        end                              
    end
end


function factor = NormaliseFactor(fea)
    factor = repmat(sqrt(sum(fea.^2,1)), size(fea, 1), 1);
end