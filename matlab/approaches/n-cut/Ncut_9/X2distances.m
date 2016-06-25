function distances = X2distances(X,Sigma);
%Timothee Cour, 2004
[n,k] = size(X);
if nargin >= 2
    X = X*inv(sqrtm(Sigma));
end
temp = sum(X.*X,2);
temp = repmat(temp,1,n);
distances = -2*X*X' + temp + temp';