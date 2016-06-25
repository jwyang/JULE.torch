function [A,B] = compute_relation2(data,scale_sig,order)
%
%      [W,Dist] = compute_relation(data,scale_sig) 
%       Input: data= Feature_dimension x Num_data
%       ouput: W = pair-wise data similarity matrix
%              Dist = pair-wise Euclidean distance
%
%
% Jianbo Shi, 1997 

if (~exist('order')),
  order = 2;
%   order = 1;
end

n = size(data,2);

B = zeros(n);
for j = 1:n
  B(:,j) = (sqrt((data(1,:)-data(1,j)).^2 +...
                (data(2,:)-data(2,j)).^2))';
end

if (~exist('scale_sig')),
    scale_sig = 0.05*max(B(:));
%     scale_sig = 0.04*max(B(:));
end

% kNN = 5;
% B2 = B;
% for j=1:n
%     [ignore,ind] = sort(B(:,j));
%     B2(ind(kNN+1:end),j) = Inf;
%     B2(ind(1:kNN),j) = B(ind(1:kNN),j) / max(B(ind(1:kNN),j)) * 0.5;
% end
% scale_sig = 1;
% B = min(B2,B2');%(B+B')/2;

tmp = (B/scale_sig).^order;

A = exp(-tmp);


