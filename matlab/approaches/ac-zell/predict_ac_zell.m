function label_pre = predict_ac_zell(feat, nClass)
%PREDICT_GDL 此处显示有关此函数的摘要
%   此处显示详细说明
% a = 100 for USPS
% z = 0.01;
K = 20;
a = 0.95;
z = 0.01;
distance_matrix = pdist2(feat, feat);
distance_matrix = distance_matrix.^2;
% Zeta
label_pre = gacCluster(distance_matrix, nClass, 'zeta', K, a, z);
end

