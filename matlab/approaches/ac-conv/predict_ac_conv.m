function label_pre = predict_ac_conv(feat, nClass)
%PREDICT_AC_CONV 此处显示有关此函数的摘要
%   此处显示详细说明
Z = linkage(feat, 'average');
label_pre = cluster(Z, 'maxClust', nClass);
end

