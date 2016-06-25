function label_pre = predict_nmf_deep(feat, nClass)
%PREDICT_NMF_DEEP �˴���ʾ�йش˺����ժҪ
%   �˴���ʾ��ϸ˵��
[Z, H, dnorm] = deep_seminmf(feat', [160 160]);
num = size(H{end}', 1);
iter = max(100, num / 100);
label_pre = litekmeans(H{end}', nClass,'MaxIter', iter, 'Replicates', 10);
end

