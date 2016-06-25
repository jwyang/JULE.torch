function [ AC, MIhat ] = evalResults( H, gnd )

nClass = length(unique(gnd));

if iscell(H)
    H = H{numel(H)};
end

label = litekmeans(H',nClass,'Replicates',100);

if(~all(size(gnd) == size(label)))
    label = label';
end

MIhat = MutualInfo(gnd,label);


res = bestMap(gnd, label);
AC = length(find(gnd == res))/length(gnd);

%disp(['Clustering AC (', num2str(k), '): ' num2str(AC), '/', num2str(MIhat)]);


end

