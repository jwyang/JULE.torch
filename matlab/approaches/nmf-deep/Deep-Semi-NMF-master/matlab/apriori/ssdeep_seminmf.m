function [ Z, H, dnorm, lambdas ] = ssdeep_seminmf ( X, layers, y, misc, varargin )

% Process optional arguments
pnames = { ...
    'z0' 'h0' 'maxiter' 'TolFun', 'verbose', 'lambdas', 'cache', ...
    'geometric'
};

num_of_layers = numel(layers);

Z = cell(1, num_of_layers);
H = cell(1, num_of_layers);
L = cell(1, num_of_layers);
W = cell(1, num_of_layers);
Di = cell(1, num_of_layers);

dflts  = {0, 0, 500, 1e-5, 1, 1, 1};

[z0, h0, maxiter, tolfun, verbose, lambdas, cache] = ...
        internal.stats.parseArgs(pnames,dflts,varargin{:});

nSmp = size(X, 2);
    
if  ~iscell(h0)
    for i_layer = 1:length(layers)
        if i_layer == 1
            V = X;
        else 
            V = H{i_layer-1};
        end
        
        if verbose
            display(sprintf('Initialising Layer #%d with k=%d with size(V)=%s...', i_layer, layers(i_layer), mat2str(size(V))));
        end
        
        if numel(y{i_layer}) == 1
            options = [];
            options.WeightMode = 'Binary';  
            options.k = y{i_layer};
            W{i_layer} = constructW(V',options)';
        else
            
            options = [];
            options.NeighborMode = 'Supervised';
            options.gnd = y{i_layer};
            options.WeightMode = 'HeatKernel';
            options.t = 1;
            W{i_layer} = constructW(V', options);
%             W{i_layer} = zeros(nSmp, nSmp);
% 
%             for j=1:nSmp;
%                 W{i_layer}(j, :) =  y{i_layer} == y{i_layer}(j);
%                 W{i_layer}(j, j) = 0;
%             end
        end

        

        W{i_layer} = lambdas(i_layer) * W{i_layer};
        DCol = full(sum(W{i_layer},2));
        Di{i_layer} = spdiags(DCol,0,nSmp,nSmp);   
   
        
        if ~iscell(z0)
%              [Z{i_layer}, H{i_layer}, ~] = ...
%              seminmf(V, ...
%                  layers(i_layer), ...
%                  'maxiter', 500, 'verbose', verbose,  'save', cache, 'fast', 1); 
        [Z{i_layer}, H{i_layer}, ~, ~, ~] = ...
             ssnmf(V, y{i_layer}, ...
                 layers(i_layer), ...
                 'maxiter', 500, 'verbose', verbose, 'lambda', lambdas(i_layer), 'save', cache); 
        else
            display('Using existing Z');
          [Z{i_layer}, H{i_layer}, ~, ~, ~] = ...
             ssnmf(V, y{i_layer}, ...
                 layers(i_layer), ...
                 'maxiter', 500, 'verbose', verbose, 'lambda', lambdas(i_layer), 'save', cache, 'z0', z0); 
        end   
            

         L{i_layer} = Di{i_layer} - W{i_layer};
    end

else
    Z=z0;
    H=h0;
    
    if verbose
        display('Skipping initialization, using provided init matrices...');
    end
end

eval(Z, H, misc);

dnorm0 = cost_function(X, Z, H, L);
dnorm = dnorm0;

if verbose
    display(sprintf('#%d error: %f', 0, dnorm0));
end

eval(Z, H, misc);


%% Error Propagation
if verbose
    display('Finetuning...');
end

for iter = 1:maxiter      
%     H_err{numel(layers)} = H{numel(layers)};
%     for i_layer = numel(layers)-1:-1:1
%         H_err{i_layer} = Z{i_layer+1} * H_err{i_layer+1};
%     end
    
    for i = 1:numel(layers)
        try
            if i == 1
                Z{i} = X  * pinv(Z{2} * H{2});
            else
                Z{i} = pinv(D') * X * pinv(H{i});
            end
        catch 
            display(sprintf('Convergance error %f. min Z{i}: %f. max %f', norm(Z{i}, 'fro'), min(min(Z{i})), max(max(Z{i})))); 
            any(isnan(Z{i}(:)))
            any(isinf(Z{i}(:)))
        end
                
        if i == 1
            D = Z{1}';
        else
            D = Z{i}' * D;
        end

        A = D * X;
        Ap = max(A, 0);
        An = -min(A, 0);

        B = D * D';

        Bp = max(B, 0);
        Bn = -min(B, 0);
        
        H{i} = H{i} .* ((Ap + Bn * H{i} + H{i} * W{i} ) ./ max(An + Bp * H{i} +  H{i} * Di{i}, 1e-6)).^0.5;        
    end
    
    assert(i == numel(layers));
    
    if mod(iter, 50) == 0
        eval(Z, H, misc);
    end   

    dnorm = cost_function(X, Z, H, L);
    
    if verbose
        if mod(iter, 10) == 0
        display(sprintf('#%d error: %f', iter, dnorm));
        end
    end
    
%     assert(dnorm <= dnorm0 + 1, ...
%         sprintf('Rec. error increasing! From %f to %f. (%d)', ...
%         dnorm0, dnorm, iter) ...
%     );
    
%     if dnorm0-dnorm <= tolfun*max(1,dnorm0) 
%         if verbose
%             display( ...
%                 sprintf('Stopped at %d: dnorm: %f, dnorm0: %f', ...
%                     iter, dnorm, dnorm0 ...
%                 ) ...
%             );
%         end
%         break;
%     end
    
    dnorm0 = dnorm;
end
end

function eval(Z, H, misc)
    names = {'Pose', 'Emotion', 'Identity'};

    for i = 1:3;
        fprintf('%s: ', names{i});
        ....
            for j = 1:3;
                try
                    %          mdl = train(misc.Y_train{i}, sparse(reshape(cell2mat(Htrain), 1200, [])'), '-q');
                    mdl = train(misc.Y_train{i}, sparse(H{j}'), '-q');
                    D = Z{1};
                    for k = 2:j
                        D = D * Z{k};
                    end

                    Hr = pinv(D) * misc.Xr;
                    [~, ac, ~] = predict(misc.Y_test{i}, sparse(Hr'), mdl, '-q');
                    fprintf(1, '%.2f | ', ac(1));
                catch 
                    
                end
            end
            fprintf(1, '\n');
    end
end


function eval(Z, H, misc)
    names = {'Pose', 'Emotion', 'Identity'};

    fprintf('\n');
    for i = 1:3;
        fprintf('%s: ', names{i});
        for j = 1:length(H);
            %mdl = train(misc.Y_train{i}, sparse(reshape(cell2mat(Htrain), 1200, [])'), '-q');
            mdl = train(misc.Y_train{i}, sparse(H{j}'), '-q');
            D = Z{1};
            for k = 2:j
                D = D * Z{k};
            end

            Hr = pinv(D) * misc.Xr;
            [~, ac, ~] = predict(misc.Y_test{i}, sparse(Hr'), mdl, '-q');

            fprintf(1, '%.2f | ', ac(1));
        end
        fprintf(1, '\n');
    end
end

function error = cost_function(X, Z, H, L)
    error = norm(X - reconstruction(Z, H), 'fro');
    
    for i = 1:length(H);
       error = error + trace(H{i} * L{i} * H{i}');
    end
end

function [ out ] = reconstruction( Z, H )

    out = H{numel(H)};

    for k = numel(H) : -1 : 1;
        out =  Z{k} * out;
    end

end
