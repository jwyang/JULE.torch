function edgemap = computeEdges(imageX,parametres,threshold)
% edgemap = computeEdges(imageX,parametres,threshold)
%
% computes the edge in imageX with parameters parametres and threshold
% Timothee Cour, Stella Yu, Jianbo Shi, 2004.

[ex,ey,egx,egy,eg_par,eg_th,emag,ephase , g ] = quadedgep(imageX,parametres,threshold);
% example : [ex,ey,egx,egy,eg_par,eg_th,emag,ephase] = quadedgep(imageX,[4,3,30,3],0.05);

% [emagTrie,eindex] = sort(emag);

%edges3 = sparse(floor(ex),floor(ey),(egx.^2+egy.^2).^(1/2),size(imageX,2),size(imageX,1))';

try
    edges2 = emag .* edge(imageX,'canny') ;
    %edges2 = emag .* edge(imageX,'sobel') ;
catch
    edges2 = 0 * emag;
end

edges2 = edges2 .* (edges2 > threshold);
egx1 = g(:,:,1);
egy1 = g(:,:,2);
eindex = find(edges2);
[ey,ex,values] = find(edges2);

egx = egx1(eindex);
egy = egy1(eindex);

edgemap.eindex = eindex;
edgemap.values = values;
edgemap.x = ex;
edgemap.y = ey;
edgemap.gx = egx;
edgemap.gy = egy;
edgemap.emag = emag;
edgemap.ephase = ephase;
edgemap.imageEdges = edges2;
