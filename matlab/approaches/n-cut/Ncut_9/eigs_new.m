function  varargout = eigs(varargin)
%EIGS  Find a few eigenvalues and eigenvectors of a matrix using ARPACK
%   D = EIGS(A) returns a vector of A's 6 largest magnitude eigenvalues.
%   A must be square and should be large and sparse.
%
%   [V,D] = EIGS(A) returns a diagonal matrix D of A's 6 largest magnitude
%   eigenvalues and a matrix V whose columns are the corresponding
%   eigenvectors.
%
%   [V,D,FLAG] = EIGS(A) also returns a convergence flag. If FLAG is 0 then
%   all the eigenvalues converged; otherwise not all converged.
%
%   EIGS(A,B) solves the generalized eigenvalue problem A*V == B*V*D. B
%   must be symmetric (or Hermitian) positive definite and the same size as
%   A. EIGS(A,[],...) indicates the standard eigenvalue problem A*V == V*D.
%
%   EIGS(A,K) and EIGS(A,B,K) return the K largest magnitude eigenvalues.
%
%   EIGS(A,K,SIGMA) and EIGS(A,B,K,SIGMA) return K eigenvalues. If SIGMA is:
%      'LM' or 'SM' - Largest or Smallest Magnitude
%   For real symmetric problems, SIGMA may also be:
%      'LA' or 'SA' - Largest or Smallest Algebraic
%      'BE' - Both Ends, one more from high end if K is odd
%   For nonsymmetric and complex problems, SIGMA may also be:
%      'LR' or 'SR' - Largest or Smallest Real part
%      'LI' or 'SI' - Largest or Smallest Imaginary part
%   If SIGMA is a real or complex scalar including 0, EIGS finds the
%   eigenvalues closest to SIGMA. For scalar SIGMA, and when SIGMA = 'SM',
%   B need only be symmetric (or Hermitian) positive semi-definite since it
%   is not Cholesky factored as in the other cases.
%
%   EIGS(A,K,SIGMA,OPTS) and EIGS(A,B,K,SIGMA,OPTS) specify options:
%   OPTS.issym: symmetry of A or A-SIGMA*B represented by AFUN [{false} |
%   true]
%   OPTS.isreal: complexity of A or A-SIGMA*B represented by AFUN [false | {true}]
%   OPTS.tol: convergence: Ritz estimate residual <= tol*NORM(A) [scalar | {eps}]
%   OPTS.maxit: maximum number of iterations [integer | {300}]
%   OPTS.p: number of Lanczos vectors: K+1<p<=N [integer | {2K}]
%   OPTS.v0: starting vector [N-by-1 vector | {randomly generated}]
%   OPTS.disp: diagnostic information display level [0 | {1} | 2]
%   OPTS.cholB: B is actually its Cholesky factor CHOL(B) [{false} | true]
%   OPTS.permB: sparse B is actually CHOL(B(permB,permB)) [permB | {1:N}]
%   Use CHOL(B) instead of B when SIGMA is a string other than 'SM'.
%
%   EIGS(AFUN,N) accepts the function AFUN instead of the matrix A. AFUN is
%   a function handle and Y = AFUN(X) should return
%      A*X            if SIGMA is unspecified, or a string other than 'SM'
%      A\X            if SIGMA is 0 or 'SM'
%      (A-SIGMA*I)\X  if SIGMA is a nonzero scalar (standard problem)
%      (A-SIGMA*B)\X  if SIGMA is a nonzero scalar (generalized problem)
%   N is the size of A. The matrix A, A-SIGMA*I or A-SIGMA*B represented by
%   AFUN is assumed to be real and nonsymmetric unless specified otherwise
%   by OPTS.isreal and OPTS.issym. In all these EIGS syntaxes, EIGS(A,...)
%   may be replaced by EIGS(AFUN,N,...).
%
%   Example:
%      A = delsq(numgrid('C',15));  d1 = eigs(A,5,'SM');
%
%   Equivalently, if dnRk is the following one-line function:
%      %----------------------------%
%      function y = dnRk(x,R,k)
%      y = (delsq(numgrid(R,k))) \ x;
%      %----------------------------%
%
%      n = size(A,1);  opts.issym = 1;
%      d2 = eigs(@(x)dnRk(x,'C',15),n,5,'SM',opts);
%
%   See also EIG, SVDS, ARPACKC, FUNCTION_HANDLE.

%   Copyright 1984-2008 The MathWorks, Inc.
%   $Revision: 1.45.4.11 $  $Date: 2008/12/01 07:19:19 $

%   EIGS provides the reverse communication interface to ARPACK library
%   routines. EIGS attempts to provide an interface for as many different
%   algorithms as possible. The reverse communication interfaces are
%   documented in the ARPACK Users' Guide, ISBN 0-89871-407-9.

cputms = zeros(5,1);
t0 = cputime; % start timing pre-processing

% Process inputs and do error-checking
if (nargout > 3)
    error('MATLAB:eigs:TooManyOutputs', 'Too many output arguments.')
end

% Platform dependent integer type
if strfind(computer, '64')
    intconvert = @(arraytoconvert) int64(arraytoconvert);
    inttype = 'int64';
else
    intconvert = @(arraytoconvert) int32(arraytoconvert);
    inttype = 'int32';
end

% Error check inputs and derive some information from them
[A,Amatrix,isrealprob,issymA,n,B,classAB,k,eigs_sigma,whch, ...
    sigma,tol,maxit,p,info,eigs_display,cholB,permB,resid,useeig, ...
    afunNargs,style,mode,Afactors,Bfactors] = ...
    checkInputs(varargin{:});

% Now have enough information to do early return on cases EIGS does not
% handle. For these cases, use the full EIG code.
if useeig
    fullEig(nargout);
    return
end

if ~isempty(Afactors)
    L = Afactors.L;     Afactors.L = [];
    U = Afactors.U;     Afactors.U = [];
    pp = Afactors.pp;   Afactors.pp = [];
    qq = Afactors.qq;   Afactors.qq = [];
    dgAsB = Afactors.dgAsB;    Afactors.dgAsB = [];
    clear Afactors;
end

if ~isempty(Bfactors)
    BisHpd = Bfactors.BisHpd;
    if BisHpd
        RB = Bfactors.RB;       Bfactors.RB = [];
        RBT = Bfactors.RBT;     Bfactors.RBT = [];
        permB = Bfactors.permB; Bfactors.permB = [];
    else
        LB = Bfactors.LB;       Bfactors.LB = [];
        UB = Bfactors.UB;       Bfactors.UB = [];
        ppB = Bfactors.ppB;     Bfactors.ppB = [];
        qqB = Bfactors.qqB;     Bfactors.qqB = [];
        dgB = Bfactors.dgB;     Bfactors.dgB = [];
    end
    clear Bfactors;
end

if isempty(B)
    if mode ~= 3
        % OP = A
        applyOP = @(v)Amtimes(v);
    else
        % OP = (A-\sigma*I)^{-1}
        applyOP = @(v)AminusSigmaBsolve(v);
    end
else
    if mode ~= 3 && BisHpd == true
        % OP = L^{-1}AL^{-T} (B = LL^T)
        applyOP = @(v)RBTsolve(Amtimes(RBsolve(v)));
    elseif mode ~= 3 && BisHpd == false
        % OP = U^{-1}L^{-1}A (B = LU)
        applyOP = @(v)Bsolve(Amtimes(v));
    else
        % OP = (A-\sigma*B)^{-1}B
        applyOP = @(v)AminusSigmaBsolve(Bmtimes(v));
    end
end
if strcmp(style,'G')
    if ~isempty(B) && BisHpd == true && mode == 3
        applyM = @(v) Bmtimes(v);
    else
        applyM = @(v) v;
    end
end

% Allocate outputs and ARPACK work variables
if isrealprob
    if issymA % real and symmetric
        if strcmp(classAB,'single')
            aupdfun = 'ssaupd';
            eupdfun = 'sseupd';
        else
            aupdfun = 'dsaupd';
            eupdfun = 'dseupd';
        end
      lworkl = intconvert(p*(p+8));
        d = zeros(k,1,classAB);
    else % real but not symmetric
        if strcmp(classAB,'single')
            aupdfun = 'snaupd';
            eupdfun = 'sneupd';
        else
            aupdfun = 'dnaupd';
            eupdfun = 'dneupd';
        end
      lworkl = intconvert(3*p*(p+2));
        workev = zeros(3*p,1,classAB);
        d = zeros(k+1,1,classAB);
        di = zeros(k+1,1,classAB);
    end
    v = zeros(n,p,classAB);
    workd = zeros(n,3,classAB);
    workl = zeros(lworkl,1,classAB);
else % complex
    if strcmp(classAB,'single')
        aupdfun = 'cnaupd';
        eupdfun = 'cneupd';
    else
        aupdfun = 'znaupd';
        eupdfun = 'zneupd';
    end
    zv = zeros(2*n*p,1,classAB);
    workd = complex(zeros(n,3,classAB));
    zworkd = zeros(2*numel(workd),1,classAB);
   lworkl = intconvert(2*(3*p^2+5*p));
   workl = zeros(lworkl,1,classAB);
    workev = zeros(2*2*p,1,classAB);
    zd = zeros(2*(k+1),1,classAB);
    rwork = zeros(p,1,classAB);
end
ldv = intconvert(n);
ipntr = zeros(15,1,inttype);
ido = intconvert(0); % reverse communication parameter, initial value
if strcmp(style,'S')
    bmat = 'I'; % standard eigenvalue problem
else
    bmat = 'G'; % generalized eigenvalue problem
end
nev = intconvert(k); % number of eigenvalues requested
ncv = intconvert(p); % number of Lanczos vectors
iparam = zeros(11,1,inttype);
% iparam(1) = ishift = 1 ensures we are never asked to handle ido=3
iparam([1 3 7]) = [1 maxit mode];
select = zeros(p,1,inttype);

% To Do: Remove this error when ARPACKC supports singles
if strcmp(classAB,'single')
    error('MATLAB:eigs:single', ...
        'EIGS does not support single precision inputs.')
end

% The ARPACK routines return to EIGS many times per each iteration but we
% only want to display the Ritz values once per iteration (if opts.disp>0).
% Keep track of whether we've displayed this iteration yet in eigs_iter.
eigs_iter = 0;

cputms(1) = cputime - t0; % end timing pre-processing

% Iterate until ARPACK's reverse communication parameter ido says to stop
while (ido ~= 99)
    
    t0 = cputime; % start timing ARPACK calls **aupd
    
    if isrealprob
        [~, ~] = arpackc( aupdfun, ido, ...
         bmat, intconvert(n), whch, nev, tol, resid, ncv, ...
            v, ldv, iparam, ipntr, workd, workl, lworkl, info );
    else
        % The FORTRAN ARPACK routine expects the complex input zworkd to have
        % real and imaginary parts interleaved, but the OP about to be
        % applied to workd expects it in MATLAB's complex representation with
        % separate real and imaginary parts. Thus we need both.
        zworkd(1:2:end-1) = real(workd);
        zworkd(2:2:end) = imag(workd);
        arpackc( aupdfun, ido, ...
         bmat, intconvert(n), whch, nev, tol, resid, ncv, ...
            zv, ldv, iparam, ipntr, zworkd, workl, lworkl, rwork, info );
        workd = reshape(complex(zworkd(1:2:end-1),zworkd(2:2:end)),[n,3]);
    end
    
    if (info < 0)
        error('MATLAB:eigs:ARPACKroutineError', ...
            'Error with ARPACK routine %s: info = %d', ...
            aupdfun,full(double(info)))
    end
    
    cputms(2) = cputms(2) + (cputime-t0); % end timing ARPACK calls **aupd
    t0 = cputime; % start timing MATLAB OP(X)
    
    % Compute which columns of workd ipntr references
    cols = checkIpntr;
    
    % The ARPACK reverse communication parameter ido tells EIGS what to do
    
    switch ido
        case -1
            workd(:,cols(2)) = applyOP(workd(:,cols(1)));
            if strcmp(style,'G') && mode ~= 3
                workd(:,cols(1)) = workd(:,cols(2));
            end
        case 1
            if strcmp(style,'G') && mode == 3 && (isempty(B) || BisHpd)
                % use with B-inner product for mode 3; see applyM
                workd(:,cols(2)) = AminusSigmaBsolve(workd(:,cols(3)));
            else
                % same work as case -1
                workd(:,cols(2)) = applyOP(workd(:,cols(1)));
                if strcmp(style,'G') && mode ~= 3
                    % use with std inner product; see applyM
                    workd(:,cols(1)) = workd(:,cols(2));
                end
            end
        case 2
            workd(:,cols(2)) = applyM(workd(:,cols(1)));
        case 99
            % ARPACK has converged
        otherwise
            error('MATLAB:eigs:UnknownIdo','Unknown ido.');
    end
    
    cputms(3) = cputms(3) + (cputime-t0); % end timing MATLAB OP(X)
    
    if eigs_display
        displayRitzValues;
    end
    
end % while (ido ~= 99)

t0 = cputime; % start timing post-processing

if (info < 0)
    error('MATLAB:eigs:ARPACKroutineError', ...
        'Error with ARPACK routine %s: info = %d',aupdfun,full(info));
end % if (info < 0)

if (nargout >= 2)
   rvec = intconvert(true); % compute eigenvectors
else
   rvec = intconvert(false); % do not compute eigenvectors
end

if isrealprob
    if issymA
        arpackc( eupdfun, rvec, 'A', select, ...
            d, v, ldv, sigma, ...
         bmat, intconvert(n), whch, nev, tol, resid, ncv, ...
            v, ldv, iparam, ipntr, workd, workl, lworkl, info );
        if strcmp(whch,'LM') || strcmp(whch,'LA')
            d = flipud(d);
            if (rvec == 1)
                v(:,1:k) = v(:,k:-1:1);
            end
        end
        if ((strcmp(whch,'SM') || strcmp(whch,'SA')) && (rvec == 0))
            d = flipud(d);
        end
    else
        % If sigma is complex, isrealprob=true and we use [c,z]neupd.
        % So use sigmar=sigma and sigmai=0 here in dneupd.
        arpackc( eupdfun, rvec, 'A', select, ...
            d, di, v, ldv, sigma, 0, workev, ...
         bmat, intconvert(n), whch, nev, tol, resid, ncv, ...
            v, ldv, iparam, ipntr, workd, workl, lworkl, info );
        d = complex(d,di);
        if rvec
            d(k+1) = [];
        else
            zind = find(d == 0);
            if isempty(zind)
                d = d(k+1:-1:2);
            else
                d(max(zind)) = [];
                d = flipud(d);
            end
        end
    end
else
    zsigma = [real(sigma); imag(sigma)];
    arpackc( eupdfun, rvec, 'A', select, ...
        zd, zv, ldv, zsigma, workev, ...
      bmat, intconvert(n), whch, nev, tol, resid, ncv, zv, ...
        ldv, iparam, ipntr, zworkd, workl, lworkl, ...
        rwork, info );
    if issymA
        d = zd(1:2:end-1);
    else
        d = complex(zd(1:2:end-1),zd(2:2:end));
    end
    v = reshape(complex(zv(1:2:end-1),zv(2:2:end)),[n p]);
end

flag = processEUPDinfo(nargin<3);

if (issymA) || (~isrealprob)
    if (nargout <= 1)
        if isrealprob
            varargout{1} = d;
        else
            varargout{1} = d(k:-1:1,1);
        end
    else
        varargout{1} = v(:,1:k);
        varargout{2} = diag(d(1:k,1));
        if (nargout >= 3)
            varargout{3} = flag;
        end
    end
else
    if (nargout <= 1)
        varargout{1} = d;
    else
        cplxd = find(di ~= 0);
        % complex conjugate pairs of eigenvalues occur together
        cplxd = cplxd(1:2:end);
        v(:,[cplxd cplxd+1]) = [complex(v(:,cplxd),v(:,cplxd+1)) ...
            complex(v(:,cplxd),-v(:,cplxd+1))];
        varargout{1} = v(:,1:k);
        varargout{2} = diag(d);
        if (nargout >= 3)
            varargout{3} = flag;
        end
    end
end

if (nargout >= 2) && (mode ~= 3) && ~isempty(B) && BisHpd == true
    varargout{1} = RBsolve(varargout{1});
end

if ~isempty(B)
    if nargout >= 2
        varargout{2} = varargout{2}/scaleB;
    else
        varargout{1} = varargout{1}/scaleB;
    end
end

if BisHpd == true && (nargout >= 2)
    vnorms = zeros(k,1);
    for ii = 1 : k
        vnorms(ii) = scaleB*(varargout{1}(:,ii)'*(Bmtimes(varargout{1}(:,ii))));
        varargout{1}(:,ii) = varargout{1}(:,ii)/sqrt(vnorms(ii));
    end
end

cputms(4) = cputime-t0; % end timing post-processing

cputms(5) = sum(cputms(1:4)); % total time

if (eigs_display == 2)
    printTimings;
end

%-------------------------------------------------------------------------%
% Nested functions
%-------------------------------------------------------------------------%

% checkInputs error checks the inputs to EIGS and also derives some
%   variables from them:
% A may be a matrix or a function applying OP.
% Amatrix is true if A is a matrix, false if A is a function.
% isrealprob is true if all of A, B and sigma are real, false otherwise.
% issymA is true if A is symmetric, false otherwise.
% n is the size of (square) A and B.
% B is [] for the standard problem. Otherwise it may be one of B, CHOL(B)
%   or CHOL(B(permB,permB)).
% classAB is single if either A or B is single, otherwise double.
% k is the number of eigenvalues to be computed.
% eigs_sigma is the value for sigma passed in by the user, 'LM' if it was
%   unspecified. eigs_sigma may be either a string or a scalar value.
% whch is the ARPACK string corresponding to eigs_sigma and mode.
% sigma is the ARPACK scalar corresponding to eigs_sigma and mode.
% tol is the convergence tolerance.
% maxit is the maximum number of iterations.
% p is the number of Lanczos vectors.
% info is the start value, initialized to 1 or 0 to indicate whether to use
% resid as the start vector or not.
% eigs_display is true if Ritz values should be displayed, false otherwise.
% cholB is true if CHOL(B) was passed in instead of B, false otherwise.
% permB may be [], otherwise it is the permutation in CHOL(B(permB,permB)).
% resid is the start vector if specified and info=1, otherwise all zero.
% useeig is true if we need to use EIG instead of ARPACK, otherwise false.
% afunNargs is the range of EIGS' varargin that are to be passed as
%   trailing parameters to the function as in afun(X,P1,P2,...).
    function [A,Amatrix,isrealprob,issymA,n,B,classAB,k, ...
            eigs_sigma,whch,sigma,tol,maxit,p,info,eigs_display,cholB,...
            permB,resid,useeig,afunNargs,style,mode,Afactors,Bfactors] = checkInputs(varargin)
        % Process inputs and do error-checking
        
        % Process the input A or the inputs AFUN and N
        % Start to derive some qualities (real, symmetric) about the problem
        if isfloat(varargin{1})
            A = varargin{1};
            Amatrix = true;
        else
            % By checking the function A with fcnchk, we can now use direct
            % function evaluation on the result, without resorting to feval
            A = fcnchk(varargin{1});
            Amatrix = false;
        end
        isrealprob = true;
        issymA = false;
        if Amatrix
            isrealprob = isreal(A);
            issymA = ishermitian(A);
            [m,n] = size(A);
            if (m ~= n)
                error('MATLAB:eigs:NonSquareMatrixOrFunction',...
                    'A must be a square matrix or a function.')
            end
        else
            n = varargin{2};
            nstr = 'Size of problem, ''n'', must be a positive integer.';
            if ~isscalar(n) || ~isreal(n)
                error('MATLAB:eigs:NonPosIntSize', nstr)
            end
            if issparse(n)
                n = full(n);
            end
            if (round(n) ~= n)
                warning('MATLAB:eigs:NonPosIntSize',['%s\n         ' ...
                    'Rounding input size.'],nstr)
                n = round(n);
            end
        end
        
        % Process the input B and derive the class of the problem.
        % Is B present in the eigs call or not?
        Bpresent = true;
        Bstr = 'Generalized matrix B must be the same size as A.';
        if (nargin < (3-Amatrix))
            B = [];
            Bpresent = false;
        else
            % Is the next input B or K?
            B = varargin{3-Amatrix};
            if ~isempty(B) % allow eigs(A,[],k,sigma,opts);
                if isscalar(B)
                    if n ~= 1
                        % this input is really K and B is not specified
                        B = [];
                        Bpresent = false;
                    else
                        % This input could be B or K.
                        % If A is scalar, then the only valid value for k is 1.
                        % So if this input is scalar, let it be B, namely
                        % eigs(4,2,...) assumes A=4, B=2, NOT A=4, k=2
                        if ~isnumeric(B)
                            error('MATLAB:eigs:BsizeMismatchA', Bstr);
                        end
                        % Unless, of course, the scalar is 1, in which case
                        % assume the that it is meant to be K.
                        if (B == 1) && ((Amatrix && nargin <= 3) || ...
                                (~Amatrix && nargin <= 4))
                            B = [];
                            Bpresent = false;
                        elseif ~isfloat(B)
                            error('MATLAB:eigs:BsizeMismatchA', Bstr);
                        end
                    end
                else
                    % B is a not a scalar.
                    if ~isfloat(B) || ~isequal(size(B),[n,n])
                        error('MATLAB:eigs:BsizeMismatchA', Bstr);
                    end
                    isrealprob = isrealprob && isreal(B);
                end
            end
        end
        % ARPACK can only handle homogeneous inputs
        if Amatrix
            classAB = superiorfloat(A,B);
            A = cast(A,classAB);
            B = cast(B,classAB);
        else
            if ~isempty(B)
                classAB = class(B);
            else
                classAB = 'double';
            end
        end
        
        % argOffset tells us where to get the eigs inputs K, SIGMA and OPTS.
        % If A is really the function afun, then it also helps us find the
        % trailing parameters in eigs(afun,n,[B],k,sigma,opts,P1,P2,...)
        % Values of argOffset:
        %  0: Amatrix is false and Bpresent is true:
        %     eigs(afun,n,B,k,sigma,opts,P1,P2,...)
        %  1: Amatrix and Bpresent are both true, or both false
        %     eigs(A,B,k,sigma,opts)
        %     eigs(afun,n,k,sigma,opts,P1,P2,...)
        %  2: Amatrix is true and Bpresent is false:
        %     eigs(A,k,sigma,opts)
        argOffset = Amatrix + ~Bpresent;
        
        if Amatrix && ((nargin - Bpresent)>4)
            error('MATLAB:eigs:TooManyInputs', 'Too many inputs.')
        end
        
        % Process the input K.
        if (nargin < (4-argOffset))
            k = min(n,6);
        else
            k = varargin{4-argOffset};
            kstr = ['Number of eigenvalues requested, k, must be a' ...
                ' positive integer <= n.'];
            if ~isnumeric(k) || ~isscalar(k) || ~isreal(k) || (k>n)
                error('MATLAB:eigs:NonIntegerEigQty', kstr)
            end
            if issparse(k)
                k = full(k);
            end
            if (round(k) ~= k)
                warning('MATLAB:eigs:NonIntegerEigQty',['%s\n         ' ...
                    'Rounding number of eigenvalues.'],kstr)
                k = round(k);
            end
        end
        
        % Process the input SIGMA and derive ARPACK values whch and sigma.
        % eigs_sigma is the value documented in the help as "SIGMA" that is
        % passed in to EIGS. eigs_sigma may be either a scalar, including 0,
        % or a string, including 'SM'.
        % In ARPACK, eigs_sigma corresponds to two variables:
        % 1.  which, called "whch" to avoid conflict with MATLAB's function
        % 2.  sigma
        % whch is always a string. sigma is always a scalar.
        % Valid combinations are shown below. Note eigs_sigma = 0/'SM' has
        % the same sigma/whch values as eigs_sigma='LM' (default) so these
        % must be distinguished by the mode.
        % eigs_sigma = 'SM' or 0 => sigma = 0, whch = 'LM' (mode=3)
        % eigs_sigma is a string not 'SM' => sigma = 0, whch = eigs_sigma (mode=1)
        % eigs_sigma is any scalar => sigma = eigs_sigma, whch = 'LM'
        % (mode=1)
        whchstr = 'Eigenvalue range sigma must be a valid 2-element string.';
        if (nargin < (5-argOffset))
            % default: eigs 'LM' => ARPACK which='LM', sigma=0
            eigs_sigma = 'LM';
            whch = 'LM';
            sigma = 0;
        else
            eigs_sigma = varargin{5-argOffset};
            if ischar(eigs_sigma)
                % eigs(string) => ARPACK which=string, sigma=0
                if ~isequal(size(eigs_sigma),[1,2])
                    error('MATLAB:eigs:EigenvalueRangeNotValid', ...
                        [whchstr '\nFor real symmetric A, the' ...
                        ' choices are ''%s'', ''%s'', ''%s'', ''%s'' or ''%s''.' ...
                        '\nFor non-symmetric or complex' ...
                        ' A, the choices are ''%s'', ''%s'', ''%s'', ''%s'',' ...
                        ' ''%s'' or ''%s''.\n'], ...
                        'LM','SM','LA','SA','BE','LM','SM','LR','SR','LI','SI')
                end
                eigs_sigma = upper(eigs_sigma);
                if strcmp(eigs_sigma,'SM')
                    % eigs('SM') => ARPACK which='LM', sigma=0
                    whch = 'LM';
                else
                    % eigs(string), where string~='SM' => ARPACK which=string, sigma=0
                    whch = eigs_sigma;
                end
                sigma = zeros(classAB);
            else
                % eigs(scalar) => ARPACK which='LM', sigma=scalar
                if ~isfloat(eigs_sigma) || ~isscalar(eigs_sigma)
                    error('MATLAB:eigs:EigenvalueShiftNonScalar',...
                        'Eigenvalue shift sigma must be a scalar.')
                end
                sigma = eigs_sigma;
                if issparse(sigma)
                    sigma = full(sigma);
                end
                sigma = cast(sigma,classAB);
                isrealprob = isrealprob && isreal(sigma);
                whch = 'LM';
            end
        end
        
        % Process the input OPTS and derive some ARPACK values.
        % ARPACK's minimum tolerance is eps/2 ([S/D]LAMCH's EPS)
        tol = eps(classAB);
        maxit = [];
        p = [];
        style = [];
        % Always use resid as the start vector, whether it is OPTS.v0 or
        % randomly generated within eigs.  We default resid to empty here.
        % If the user does not initialize it, we provide a random residual
        % below.
      info = intconvert(1);
        resid = [];
        eigs_display = 1;
        cholB = false; % do we have B or its Cholesky factor?
        permB = []; % if cholB, is it chol(B), or chol(B(permB,permB))?
        if (nargin >= (6-argOffset))
            opts = varargin{6-argOffset};
            if ~isa(opts,'struct')
                error('MATLAB:eigs:OptionsNotStructure',...
                    'Options argument must be a structure.')
            end
            if isfield(opts,'issym') && ~Amatrix
                issymA = opts.issym;
                if (issymA ~= false) && (issymA ~= true)
                    error('MATLAB:eigs:InvalidOptsIssym', ...
                        'opts.issym must be true or false.')
                end
            end
            if isfield(opts,'isreal') && ~Amatrix
                if (opts.isreal ~= false) && (opts.isreal ~= true)
                    error('MATLAB:eigs:InvalidOptsIsreal', ...
                        'opts.isreal must be true or false.')
                end
                isrealprob = isrealprob && opts.isreal;
            end
            if ~isempty(B) && (isfield(opts,'cholB') || isfield(opts,'permB'))
                if isfield(opts,'cholB')
                    cholB = opts.cholB;
                    if (cholB ~= false) && (cholB ~= true)
                        error('MATLAB:eigs:InvalidOptsCholB', ...
                            'opts.cholB must be true or false.')
                    end
                    if isfield(opts,'permB')
                        if issparse(B) && cholB
                            permB = opts.permB;
                            if ~isvector(permB) || ~isequal(sort(permB(:)),(1:n)')
                                error('MATLAB:eigs:InvalidOptsPermB',...
                                    'opts.permB must be a permutation of 1:n.')
                            end
                        else
                            warning('MATLAB:eigs:IgnoredOptionPermB', ...
                                ['Ignoring opts.permB since B is not its sparse' ...
                                ' Cholesky factor.'])
                        end
                    end
                end
            end
            if isfield(opts,'tol')
                if ~isfloat(tol) || ~isscalar(opts.tol) || ~isreal(opts.tol) || (opts.tol<=0)
                    error('MATLAB:eigs:InvalidOptsTol',...
                        ['Convergence tolerance opts.tol must be a strictly' ...
                        ' positive real scalar.'])
                end
                tol = cast(full(opts.tol),classAB);
            end
            if isfield(opts,'p')
                p = opts.p;
                pstr = ['Number of basis vectors opts.p must be a positive' ...
                    ' integer <= n.'];
                if ~isnumeric(p) || ~isscalar(p) || ~isreal(p) || (p<=0) || (p>n)
                    error('MATLAB:eigs:InvalidOptsP', pstr)
                end
                if issparse(p)
                    p = full(p);
                end
                if (round(p) ~= p)
                    warning('MATLAB:eigs:NonIntegerVecQty',['%s\n         ' ...
                        'Rounding number of basis vectors.'],pstr)
                    p = round(p);
                end
            end
            if isfield(opts,'maxit')
                maxit = opts.maxit;
                str = ['Maximum number of iterations opts.maxit must be' ...
                    ' a positive integer.'];
                if ~isnumeric(maxit) || ~isscalar(maxit) || ~isreal(maxit) || (maxit<=0)
                    error('MATLAB:eigs:OptsMaxitNotPosInt', str)
                end
                if issparse(maxit)
                    maxit = full(maxit);
                end
                if (round(maxit) ~= maxit)
                    warning('MATLAB:eigs:NonIntegerIterationQty',['%s\n         ' ...
                        'Rounding number of iterations.'],str)
                    maxit = round(maxit);
                end
            end
            if isfield(opts,'v0')
                if ~isfloat(opts.v0) || ~isequal(size(opts.v0),[n,1])
                    error('MATLAB:eigs:WrongSizeOptsV0',...
                        'Start vector opts.v0 must be n-by-1.')
                end
                if isrealprob
                    if ~isreal(opts.v0)
                        error('MATLAB:eigs:NotRealOptsV0',...
                            'Start vector opts.v0 must be real for real problems.')
                    end
                    resid(1:n,1) = full(opts.v0);
                else
                    resid(2:2:2*n,1) = full(imag(opts.v0));
                    resid(1:2:(2*n-1),1) = full(real(opts.v0));
                end
            end
            if isfield(opts,'disp')
                eigs_display = opts.disp;
                dispstr = 'Diagnostic level opts.disp must be an integer.';
                if ~isnumeric(eigs_display) || ~isscalar(eigs_display) || ...
                        ~isreal(eigs_display) || (eigs_display<0)
                    error('MATLAB:eigs:NonIntegerDiagnosticLevel', dispstr)
                end
                if (round(eigs_display) ~= eigs_display)
                    warning('MATLAB:eigs:NonIntegerDiagnosticLevel', ...
                        '%s\n         Rounding diagnostic level.',dispstr)
                    eigs_display = round(eigs_display);
                end
            end
            if isfield(opts,'cheb')
                error('MATLAB:eigs:ObsoleteOptionCheb', ...
                    'Polynomial acceleration opts.cheb is an obsolete option.');
            end
            if isfield(opts,'stagtol')
                error('MATLAB:eigs:ObsoleteOptionStagtol', ...
                    'Stagnation tolerance opts.stagtol is an obsolete option.');
            end
            if isfield(opts,'style')
                style = opts.style;
                str = 'style must be ''S'' or ''G''.';
                if ~ischar(style)
                    error('MATLAB:eigs:InvalidStyle',str);
                elseif ~(strcmp(style,'S') || strcmp(style,'G'))
                    error('MATLAB:eigs:InvalidStyle',str);
                end
            end
        end
        if (isempty(resid))
            if isrealprob
                resid = cast(rand(n,1),classAB);
            else
                resid = cast(rand(2*n,1),classAB);
            end
        end
        
        afunNargs = zeros(1,0);
        if ~Amatrix
            % The trailing parameters for afun start at varargin{7-argOffset}
            % in eigs(afun,n,[B],k,sigma,opts,P1,P2,...). If there are no
            % trailing parameters in eigs, then afunNargs is a 1-by-0 empty
            % and no trailing parameters are passed to afun(x)
            afunNargs = 7-argOffset:nargin;
        end
        
        % Now that OPTS has been processed, do final error checking and
        % assign ARPACK variables
        
        % Extra check on input B
        if ~isempty(B)
            % B must be symmetric (Hermitian) positive (semi-)definite
            if cholB
                if ~isequal(triu(B),B)
                    error('MATLAB:eigs:BNotChol', ...
                        'opts.CHOLB specified, but B is not upper triangular.');
                end
            end
        end
        
        if isempty(style)
            if strcmp(eigs_sigma,'SM') || isscalar(eigs_sigma) || ~isempty(B)
                style = 'G';
            else
                style = 'S';
            end
        end
        
        if ~isempty(B)
            scaleB = norm(B,'fro')/sqrt(n);
            scaleB = 2^(floor(log2(scaleB+1)));
            B = B/scaleB;
            if cholB
                scaleB = scaleB^2;
            end
            if isscalar(eigs_sigma)
                sigma = scaleB*eigs_sigma;
            end
        end
        
        
        if strcmp(eigs_sigma,'SM') || ~ischar(eigs_sigma)
            % eigs(A,B,k,scalarSigma) or eigs(A,B,k,'SM'), B may be []
            % Note: sigma must be real for [s,d]saupd and [s,d]naupd
            % If sigma is complex, even if A and B are both real, we use
            % [c,z]naupd.
            % This means that mode=3 in [s,d]naupd, which has
            % OP = real(inv(A - sigma*M)*M) and B = M
            % reduces to the same OP as [s,d]saupd and [c,z]naupd.
            % A*x = lambda*M*x, M symmetric (positive) semi-definite
            % => OP = inv(A - sigma*M)*M and B = M
            % => shift-and-invert mode
            mode = 3;
        elseif strcmp(style,'S')
            % eigs(A,k,stringSigma) or eigs(A,[],k,stringSigma),
            % stringSigma~='SM'
            % A*x = lambda*x
            % => OP = A and B = I
            mode = 1;
        else
            % eigs(A,B,k,stringSigma), stringSigma~='SM'
            % A*x = lambda*B*x
            % => OP = inv(B)*A and use standard inner product.
            mode = 2;
        end
        
        BisHpd = false;
        if cholB || (~isempty(B) && ishermitian(B))
            % The reordering permutation permB is [] unless B is sparse
            [RB,RBT,permB,BisHpd] = CHOLfactorB;
            if mode == 3 && ~cholB
                RB = [];    RBT = [];   permB = [];
            end
        end
        
        qqB = [];
        if BisHpd == false && (mode == 1 || mode == 2)
            [LB,UB,ppB,qqB,dgB] = LUfactorB;
        end
        
        Bfactors = [];
        if ~isempty(B)
            if BisHpd == true
                Bfactors = struct('RB',RB,'RBT',RBT,'permB',permB,'BisHpd',BisHpd);
            elseif (mode == 1 || mode == 2)
                Bfactors = struct('LB',LB,'UB',UB,'ppB',ppB,'qqB',qqB,'dgB',dgB,'BisHpd',BisHpd);
            end
        end
        
        Afactors = [];
        qq = [];
        if (mode == 3) && Amatrix % need lu(A-sigma*B)
            % The reordering permutation permAsB is [] unless A-sigma*B is sparse
            [L,U,pp,qq,dgAsB] = LUfactorAminusSigmaB;
            Afactors = struct('L',L,'U',U,'pp',pp,'qq',qq,'dgAsB',dgAsB);
        end % if (mode == 3) && Amatrix        
        
        % under these conditions, OP must be unsymmetric
        % note that OP = inv(A-\sigma*B)*B IS symmetric if A is symmetric
        % and B-inner product is used!
        if ~isempty(B) && (BisHpd == false || (strcmp(style,'S') && mode == 3))
            issymA = false;
        end
        % Extra check on input K
        % We fall back on using the full EIG code if K is too large.
        useeig = false;
        if isrealprob && issymA
            knstr = sprintf(['For real symmetric problems, must have' ...
                ' number of eigenvalues k < n.\n']);
        else
            knstr = sprintf(['For nonsymmetric and complex problems,' ...
                ' must have number of eigenvalues k < n-1.\n']);
        end
        if isempty(B)
            knstr = [knstr 'Using eig(full(A)) instead.'];
        else
            knstr = [knstr 'Using eig(full(A),full(B)) instead.'];
        end
        if (k == 0)
            useeig = true;
        end
        if isrealprob && issymA
            if (k > n-1)
                if (n >= 6)
                    warning('MATLAB:eigs:TooManyRequestedEigsForRealSym', ...
                        '%s',knstr)
                end
                useeig = true;
            end
        else
            if (k > n-2)
                if (n >= 7)
                    warning('MATLAB:eigs:TooManyRequestedEigsForComplexNonsym', ...
                        '%s',knstr)
                end
                useeig = true;
            end
        end
        
        % Extra check on input SIGMA
        if isrealprob && issymA
            if ~isreal(sigma)
                error('MATLAB:eigs:ComplexShiftForRealSymProblem',...
                    ['For real symmetric problems, eigenvalue shift sigma must' ...
                    ' be real.'])
            end
        else
            if ~isrealprob && issymA && ~isreal(sigma)
                warning('MATLAB:eigs:ComplexShiftForHermitianProblem', ...
                    ['Complex eigenvalue shift sigma on a Hermitian problem' ...
                    ' (all real eigenvalues).'])
            end
        end
        if isrealprob && issymA
            if strcmp(whch,'LR')
                whch = 'LA';
                warning('MATLAB:eigs:SigmaChangedToLA', ...
                    ['For real symmetric problems, sigma value ''LR''' ...
                    ' (Largest Real) is now ''LA'' (Largest Algebraic).'])
            end
            if strcmp(whch,'SR')
                whch = 'SA';
                warning('MATLAB:eigs:SigmaChangedToSA', ...
                    ['For real symmetric problems, sigma value ''SR''' ...
                    ' (Smallest Real) is now ''SA'' (Smallest Algebraic).'])
            end
            if ~ismember(whch,{'LM', 'SM', 'LA', 'SA', 'BE'})
                error('MATLAB:eigs:EigenvalueRangeNotValid', ...
                    [whchstr '\nFor real symmetric A, the' ...
                    ' choices are ''%s'', ''%s'', ''%s'', ''%s'' or ''%s''.'], ...
                    'LM','SM','LA','SA','BE');
            end
        else
            if strcmp(whch,'BE')
                warning('MATLAB:eigs:SigmaChangedToLM', ...
                    ['Sigma value ''BE'' is now only available for real' ...
                    ' symmetric problems.  Computing ''LM'' eigenvalues instead.'])
                whch = 'LM';
            end
            if ~ismember(whch,{'LM', 'SM', 'LR', 'SR', 'LI', 'SI'})
                error('MATLAB:eigs:EigenvalueRangeNotValid', ...
                    [whchstr '\nFor non-symmetric or complex' ...
                    ' A, the choices are ''%s'', ''%s'', ''%s'', ''%s'',' ...
                    ' ''%s'' or ''%s''.\n'],'LM','SM','LR','SR','LI','SI');
            end
        end
        
        % The remainder of the error checking does not apply for the large
        % values of K that force us to use full EIG instead of ARPACK.
        if useeig
            return
        end
        
        % Extra check on input OPTS.p
        if isempty(p)
            if isrealprob && ~issymA
                p = min(max(2*k+1,20),n);
            else
                p = min(max(2*k,20),n);
            end
        else
            if isrealprob && issymA
                if (p <= k)
                    error('MATLAB:eigs:InvalidOptsPforRealSymProb',...
                        ['For real symmetric problems, must have number of' ...
                        ' basis vectors opts.p > k.'])
                end
            else
                if (p <= k+1)
                    error('MATLAB:eigs:InvalidOptsPforComplexOrNonSymProb',...
                        ['For nonsymmetric and complex problems, must have number of' ...
                        ' basis vectors opts.p > k+1.'])
                end
            end
        end
        
        % Extra check on input OPTS.maxit
        if isempty(maxit)
            maxit = max(300,ceil(2*n/max(p,1)));
        end
        
        
        % Nested functions in checkInputs
        %-------------------------------------------------------------------------%
        function [RB,RBT,ppB,BisHpd] = CHOLfactorB
            % permB may be [] (from checkInputs) if the problem is not sparse
            % or if it was not passed in as opts.permB
            ppB = permB;
            if cholB
                % CHOL(B) was passed in as B
                RB = B;
                RBT = B';
                BisHpd = true;
            else
                % CHOL(B) was not passed into EIGS
                if ~isempty(B)
                    % Algorithm requires CHOL(B) to be computed
                    if issparse(B)
                        ppB = symamd(B);
                        [RB,idxB] = chol(B(ppB,ppB));
                    else
                        [RB,idxB] = chol(B);
                    end
                    if mode == 3
                        RB = [];    ppB = [];
                    end
                    RBT = RB';
                    if (idxB == 0)
                        BisHpd = true;
                    elseif mode == 3 && isreal(B)
                        % LDL decomposition is only for 'SM' eigenvalues of the
                        % pair (A,B) where B is Hermitian positive
                        % semi-definite; in this case, as ARPACK users' guide
                        % suggests, one should still use B-(semi)inner product
                        [LB,DB,pvB] = ldl(B,'vector'); %#ok
                        BisHpd = checkTridiagForHSD(diag(DB), diag(DB,1));
                    else
                        BisHpd = false;
                    end
                end
            end
            if ~isempty(B) && issparse(B)
               tmp = speye(length(B));               
               ppB = tmp(ppB,1:length(B));
            end
        end % CHOLfactorB
        %-------------------------------------------------------------------------%
        function [LB,UB,ppB,qqB,dgB] = LUfactorB
            % LU factor B, including a reordering perm if it is sparse
            if issparse(B)
                [LB,UB,ppB,qqB,dgB] = lu(B);
            else
                [LB,UB,ppB] = lu(B,'vector');
                qqB = []; dgB = [];
            end            
            % Warn if lu(B) is ill-conditioned
            dUB = diag(UB);
            if any(dUB == 0) || any(diag(LB) == 0)
                error('MATLAB:eigs:SingularB', ...
                    ['B is singular.\nUnable to compute the specified eigenvalues'...
                     ' because infinite eigenvalue(s) exist']);
            end
            rcondestUB = full(min(abs(dUB)) / max(abs(dUB)));
            if (rcondestUB < eps)
                warning('MATLAB:eigs:IllConditionedB',...
                    ['B has small reciprocal condition' ...
                    ' estimate: %f\n' ...
                    '         indicating that B^{-1}A might not be applied accurately.\n' ...
                    '         and there may be infinite eigenvalue(s).\n'],...
                    rcondestUB);
            end
        end % LUfactorB
        
        %-------------------------------------------------------------------------%
        function [L,U,pp,qq,dgAsB] = LUfactorAminusSigmaB
            % LU factor A-sigma*B, including a reordering perm if it is sparse
            if sigma == 0
                AsB = A;
            elseif isempty(B)
                if issparse(A)
                    AsB = A - sigma * speye(n);
                else
                    AsB = A - sigma * eye(n);
                end
            else
                if cholB
                    if issparse(B)
                        AsB = A - sigma * checkInputBmtimes(speye(n));
                    else
                        AsB = A - sigma * checkInputBmtimes(eye(n));
                    end
                else
                    AsB = A - sigma * B;
                end
            end
            if issparse(AsB)
                [L,U,pp,qq,dgAsB] = lu(AsB);
            else
                [L,U,pp] = lu(AsB,'vector');
                qq = []; dgAsB = [];
            end
            % Warn if lu(A-sigma*B) is ill-conditioned
            % => sigma is close to an exact eigenvalue of (A,B)
            if isempty(B)
                ds = '(A-sigma*I)';
            else
                ds = '(A-sigma*B)';
            end
            dU = diag(U);
            if any(dU == 0) || any(diag(L) == 0)
                error('MATLAB:eigs:AminusBSingular', ...
                    [ds 'is singular. The shift is an eigenvalue.\n' ...
                    'Try to use some other shift please.\n']);
            end
            rcondestU = full(min(abs(dU)) / max(abs(dU)));
            if (rcondestU < eps)                
                warning('MATLAB:eigs:SigmaNearExactEig',...
                    [ds ' has small reciprocal condition' ...
                    ' estimate: %f\n' ...
                    '         indicating that sigma is near an exact' ...
                    ' eigenvalue.\n         The algorithm may not converge unless' ...
                    ' you try a new value for sigma.\n'], ...
                    rcondestU);
            end
        end % LUfactorAminusSigmaB
        
%-------------------------------------------------------------------------%        
        function v = checkInputBmtimes(u)
            % Matrix-vector multiply v = B*u
            if cholB % use B's cholesky factor and its transpose
                if ~isempty(permB)
                    v = permB'*(RBT * (RB * (permB*u)));
                else
                    v = RBT * (RB * u);
                end
            else
                v = B * u;
            end
        end
        
    end % checkInputs

%-------------------------------------------------------------------------%
    function fullEig(nOutputs)
        % Use EIG(FULL(A)) or EIG(FULL(A),FULL(B)) instead of ARPACK
        if ~isempty(B)
            B = Bmtimes(eye(n));
            B = B*scaleB;
        end
        if isfloat(A)
            if issparse(A);
                A = full(A);
            end
        else
            % A is specified by a function.
            % Form the matrix A by applying the function.
            if ischar(eigs_sigma) && ~strcmp(eigs_sigma,'SM')
                % A is a function multiplying A*x
                AA = eye(n);
                for i = 1:n
                    AA(:,i) = A(AA(:,i),varargin{afunNargs});
                end
                A = AA;
            else
                if (isfloat(eigs_sigma) && eigs_sigma == 0) || strcmp(eigs_sigma,'SM')
                    % A is a function solving A\x
                    invA = eye(n);
                    for i = 1:n
                        invA(:,i) = A(invA(:,i),varargin{afunNargs});
                    end
                    A = eye(n) / invA;
                else
                    % A is a function solving (A-sigma*B)\x
                    % B may be [], indicating the identity matrix
                    % U = (A-sigma*B)\sigma*B
                    % => (A-sigma*B)*U = sigma*B
                    % => A*U = sigma*B(U + eye(n))
                    % => A = sigma*B(U + eye(n)) / U
                    if isempty(B)
                        sB = eigs_sigma*eye(n);
                    else
                        sB = eigs_sigma*B;
                    end
                    U = zeros(n,n);
                    for i = 1:n
                        U(:,i) = A(sB(:,i),varargin{afunNargs});
                    end
                    A = sB*(U+eye(n)) / U;
                end
            end
        end
        
        if isempty(B)
            eigInputs = {A};
        else
            eigInputs = {A,B};
        end
        % Now with full floating point matrices A and B, use EIG:
        if (nOutputs <= 1)
            d = eig(eigInputs{:});
        else
            [V,D] = eig(eigInputs{:});
            d = diag(D);
        end
        
        % Grab the eigenvalues we want, based on sigma
        firstKindices = 1:k;
        lastKindices = n:-1:n-k+1;
        if ischar(eigs_sigma)
            switch eigs_sigma
                case 'LM'
                    [ignore,ind] = sort(abs(d));
                    range = lastKindices;
                case 'SM'
                    [ignore,ind] = sort(abs(d));
                    range = firstKindices;
                case 'LA'
                    [ignore,ind] = sort(d);
                    range = lastKindices;
                case 'SA'
                    [ignore,ind] = sort(d);
                    range = firstKindices;
                case 'LR'
                    [ignore,ind] = sort(abs(real(d)));
                    range = lastKindices;
                case 'SR'
                    [ignore,ind] = sort(abs(real(d)));
                    range = firstKindices;
                case 'LI'
                    [ignore,ind] = sort(abs(imag(d)));
                    range = lastKindices;
                case 'SI'
                    [ignore,ind] = sort(abs(imag(d)));
                    range = firstKindices;
                case 'BE'
                    [ignore,ind] = sort(abs(d));
                    range = [1:floor(k/2), n-ceil(k/2)+1:n];
                otherwise
                    error('MATLAB:eigs:fullEigSigma','Unknown value of sigma');
            end
        else
            % sigma is a scalar
            [ignore,ind] = sort(abs(d-eigs_sigma));
            range = 1:k;
        end
        
        if (nOutputs <= 1)
            varargout{1} = d(ind(range));
        else
            varargout{1} = V(:,ind(range));
            varargout{2} = D(ind(range),ind(range));
            if (nOutputs == 3)
                % flag indicates "convergence"
                varargout{3} = 0;
            end
        end
        
    end % FULLEIG


%-------------------------------------------------------------------------%
    function cols = checkIpntr
        % Check that ipntr returned from ARPACK refers to the start of a
        % column of workd.
        if (ido == 1) && (mode == 3) && strcmp(style,'G') 
            inds = double(ipntr(1:3));
        else
            inds = double(ipntr(1:2));
        end
        rows = mod(inds-1,n)+1;
        cols = (inds-rows)/n+1;
        if ~all(rows==1)
            error('MATLAB:eigs:ipntrMismatchWorkdColumn', ...
                ['One of ipntr(1:3) does not refer to the start' ...
                ' of a column of the %d-by-3 array workd.'],n)
        end
    end % checkIpntr

%-------------------------------------------------------------------------%
    function v = Amtimes(u)
        % Matrix-vector multiply v = A*u
        if Amatrix
            v = A * u;
        else % A is a function
            v = A(u,varargin{afunNargs});
            if isrealprob && ~isreal(v)
                error('MATLAB:eigs:complexFunction', ...
                    'AFUN is complex; set opts.isreal = false.');
            end
        end
    end

%-------------------------------------------------------------------------%
    function v = Bmtimes(u)
        % Matrix-vector multiply v = B*u
        if cholB % use B's cholesky factor and its transpose
            if ~isempty(permB)
                v = permB'*(RBT * (RB * (permB*u)));
            else
                v = RBT * (RB * u);
            end
        else
            v = B * u;
        end
    end

%-------------------------------------------------------------------------%
    function v = RBsolve(u)
        % Solve v = RB\u for v
        if issparse(B)
            if ~isempty(permB)
                v = permB'*(RB \ u);
            else
                v = RB \ u;
            end
        else
            RBopts.UT = true;
            v = linsolve(RB,u,RBopts);
        end
    end

%-------------------------------------------------------------------------%
    function v = RBTsolve(u)
        % Solve v = RB'\u for v
        if issparse(B)
            if ~isempty(permB)
                v = RBT \ (permB*u);
            else
                v = RBT \ u;
            end
        else
            RBTopts.LT = true;
            v = linsolve(RBT,u,RBTopts);
        end
    end

%-------------------------------------------------------------------------%
    function v = AminusSigmaBsolve(u)
        % Solve v = (A-sigma*B)\u for v
        if Amatrix
            if ~isempty(dgAsB)
                % use LU reordering permAsB
                v = qq*(U \ (L \ (pp*(dgAsB\u))));
            else
                v = U \ (L \ u(pp));
            end
        else % A is a function
            v = A(u,varargin{afunNargs});
            if isrealprob && ~isreal(v)
                error('MATLAB:eigs:complexFunction', ...
                    'AFUN is complex; set opts.isreal = false.');
            end
        end
    end % AminusSigmaBsolve
%-------------------------------------------------------------------------%
    function v = Bsolve(u)
        % Solve v = (A-sigma*B)\u for v
        if ~isempty(dgB)
            % use LU reordering permAsB
            v = qqB*(UB \ (LB \ (ppB*(dgB\u))));
        else
            v = UB \ (LB \ u(ppB));
        end
    end % AminusSigmaBsolve

%-------------------------------------------------------------------------%
    function displayRitzValues
        % Display a few Ritz values at the current iteration
        iter = double(ipntr(15));
        if (iter > eigs_iter) && (ido ~= 99)
            eigs_iter = iter;
            ds = sprintf(['Iteration %d: a few Ritz values of the' ...
                ' %d-by-%d matrix:'],iter,p,p);
            disp(ds)
            if isrealprob
                if issymA
                    dispvec = workl(double(ipntr(6))+(0:p-1));
                    if strcmp(whch,'BE')
                        % roughly k Large eigenvalues and k Small eigenvalues
                        disp(dispvec(max(end-2*k+1,1):end))
                    else
                        % k eigenvalues
                        disp(dispvec(max(end-k+1,1):end))
                    end
                else
                    dispvec = complex(workl(double(ipntr(6))+(0:p-1)), ...
                        workl(double(ipntr(7))+(0:p-1)));
                    % k+1 eigenvalues (keep complex conjugate pairs together)
                    disp(dispvec(max(end-k,1):end))
                end
            else
                dispvec = complex(workl(2*double(ipntr(6))-1+(0:2:2*(p-1))), ...
                    workl(2*double(ipntr(6))+(0:2:2*(p-1))));
                disp(dispvec(max(end-k+1,1):end))
            end
        end
    end

%-------------------------------------------------------------------------%
    function flag = processEUPDinfo(warnNonConvergence)
        % Process the info flag returned by the ARPACK routine **eupd
        flag = 0;
        if (info ~= 0)
            es = ['Error with ARPACK routine ' eupdfun ':\n'];
            switch double(info)
                case 2
                    ss = sum(select);
                    if (ss < k)
                        error('MATLAB:eigs:ARPACKroutineError02ssLTk', ...
                            [es 'The logical variable select was only set' ...
                            ' with %d 1''s instead of nconv=%d (k=%d).\n' ...
                            'Please report this to the ARPACK authors at' ...
                            ' arpack@caam.rice.edu.'], ...
                            ss,double(iparam(5)),k)
                    else
                        error('MATLAB:eigs:ARPACKroutineError02', ...
                            [es 'The LAPACK reordering routine %strsen' ...
                            ' did not return all %d eigenvalues.'], ...
                            aupdfun(1),k);
                    end
                case 1
                    error('MATLAB:eigs:ARPACKroutineError01', ...
                        [es 'The Schur form could not be reordered by the' ...
                        ' LAPACK routine %strsen.\nPlease report this to the' ...
                        ' ARPACK authors at arpack@caam.rice.edu.'], ...
                        aupdfun(1))
                case -14
                    error('MATLAB:eigs:ARPACKroutineErrorMinus14', ...
                        [es aupdfun ...
                        ' did not find any eigenvalues to sufficient accuracy.']);
                otherwise
                    error('MATLAB:eigs:ARPACKroutineError', ...
                        [es 'info = %d. Please consult the ARPACK Users''' ...
                        ' Guide for more information.'],full(info));
            end
        else
            nconv = double(iparam(5));
            if (nconv == 0)
                if (warnNonConvergence)
                    warning('MATLAB:eigs:NoEigsConverged', ...
                        'None of the %d requested eigenvalues converged.',k)
                else
                    flag = 1;
                end
            elseif (nconv < k)
                if (warnNonConvergence)
                    warning('MATLAB:eigs:NotAllEigsConverged', ...
                        'Only %d of the %d requested eigenvalues converged.', ...
                        nconv,k)
                else
                    flag = 1;
                end
            end
        end
    end % processEUPDinfo

%-------------------------------------------------------------------------%
    function printTimings
        % Print the time taken for each major stage of the EIGS algorithm
        if (mode == 1)
            innerstr = sprintf(['Compute A*X:' ...
                '                               %f\n'],cputms(3));
        elseif (mode == 3)
            if isempty(B)
                innerstr = sprintf(['Solve (A-SIGMA*I)*X=Y for X:' ...
                    '               %f\n'],cputms(3));
            else
                innerstr = sprintf(['Solve (A-SIGMA*B)*X=B*Y for X:' ...
                    '             %f\n'],cputms(3));
            end
        end
        if ((mode == 3) && (Amatrix))
            if isempty(B)
                prepstr = sprintf(['Pre-processing, including lu(A-sigma*I):' ...
                    '   %f\n'],cputms(1));
            else
                prepstr = sprintf(['Pre-processing, including lu(A-sigma*B):' ...
                    '   %f\n'],cputms(1));
            end
        else
            prepstr = sprintf(['Pre-processing:' ...
                '                            %f\n'],cputms(1));
        end
        sstr = sprintf('***********CPU Timing Results in seconds***********');
        ds = sprintf(['\n' sstr '\n' ...
            prepstr ...
            'ARPACK''s %s:                           %f\n' ...
            innerstr ...
            'Post-processing with ARPACK''s %s:      %f\n' ...
            '***************************************************\n' ...
            'Total:                                     %f\n' ...
            sstr '\n'], ...
            aupdfun,cputms(2),eupdfun,cputms(4),cputms(5));
        disp(ds)
    end % printTimings

%-------------------------------------------------------------------------%
% End of nested functions
%-------------------------------------------------------------------------%

end % EIGS

%-------------------------------------------------------------------------%
% Subfunctions
%-------------------------------------------------------------------------%
function tf = ishermitian(A)
%ISHERMITIAN
tf = isequal(A,A');
end % ishermititan

function BisHpd = checkTridiagForHSD(alpha, beta)
% CHECKTRIDIAGFORHSD
%   Uses Sturm sequence on alpha (diagonal) and beta (superdiagonal) to
%   determine if the matrix diag(alpha,0) + diag(beta,1) + diag(beta,-1) is
%   Positive Semi-definite.
n = length(alpha);
BisHpd = true;
d = alpha(1);
if d < 0
    BisHpd = false;
    return;
end
for k = 1:(n-1)
    if d == 0
        d = eps*(abs(beta(k))+eps);
    end
    d = alpha(k+1) - beta(k)*(beta(k)/d);
    if d < 0
        BisHpd = false;
        return;
    end
end
end % checkTridiagForHSD
%-------------------------------------------------------------------------%
% End of subfunctions
%-------------------------------------------------------------------------%
