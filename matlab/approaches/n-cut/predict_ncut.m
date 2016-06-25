function label_pre = predict_ncut(feat, nClass)
%PREDICT_NCUT 此处显示有关此函数的摘要
%   此处显示详细说明
% compute similarity matrix
feat = feat';
[W, ~] = compute_relation(feat);
tic;
[NcutDiscrete, ~, ~] = ncutW(W, nClass);
disp(['The computation took ' num2str(toc) ' seconds']);

label_pre = zeros(1, size(feat, 2));
for j = 1:nClass
    label_pre(NcutDiscrete(:,j) == 1) = j;
end

end

