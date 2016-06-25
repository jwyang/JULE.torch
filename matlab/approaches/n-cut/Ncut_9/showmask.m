function RGB=showmask(V,M,display_flag);
% showmask(V,M);
%
% M is a nonneg. mask
% Jianbo Shi, 1997

V=V-min(V(:));
V=V/max(V(:));
V=.25+0.75*V; %brighten things up a bit

M=M-min(M(:));
M=M/max(M(:));

H=0.0+zeros(size(V));
S=M;
RGB=hsv2rgb(H,S,V);

%if nargin>2
   image(RGB)
   axis('image')
%end
