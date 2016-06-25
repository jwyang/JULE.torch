function W = computeW(imageX,dataW,emag,ephase)
% W = computeW(imageX,dataW,emag,ephase)
% Timothee Cour, Stella Yu, Jianbo Shi, 2004.
[p,q] = size(imageX);

[w_i,w_j] = cimgnbmap([p,q],dataW.sampleRadius,dataW.sample_rate);

W = affinityic(emag,ephase,w_i,w_j,max(emag(:)) * dataW.edgeVariance);
W = W/max(W(:));
