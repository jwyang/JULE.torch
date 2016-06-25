function label_pre = predict_kmeans(feat, nClass)
%PREDICT_KMEANS 此处显示有关此函数的摘要
%   此处显示详细说明
num = size(feat, 1);
iter = max(100, num / 100);
% opts = statset('MaxIter', iter, 'UseParallel', true);
% label_pre = kmeans(feat, nClass, 'Options', opts);
label_pre = litekmeans(feat, nClass,'MaxIter', iter, 'Replicates', 10);
end

