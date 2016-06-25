function inKcCluster = gacFindKcCluster (affinityTab, Kc)

Kc = ceil(1.2*Kc);
% input should be symmetric
% [sortedAff, ~] = sort(affinityTab);
[sortedAff, ~] = gacMink(affinityTab, Kc, 1);
inKcCluster = bsxfun(@le, affinityTab, sortedAff(Kc,:));
inKcCluster = inKcCluster | inKcCluster';

end