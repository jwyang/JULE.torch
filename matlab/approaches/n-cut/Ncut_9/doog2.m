function G=doog2(sig,r,th,N);
% G=doog2(sig,r,th,N);
% Make difference of offset gaussians kernel
% theta is in degrees
% (see Malik & Perona, J. Opt. Soc. Amer., 1990)
%
% Example:
% >> imagesc(doog2(1,12,0,64,1))
% >> colormap(gray)
%
% Serge Belongie


no_pts=N;  % no. of points in x,y grid

[x,y]=meshgrid(-(N/2)+1/2:(N/2)-1/2,-(N/2)+1/2:(N/2)-1/2);

phi=pi*th/180;
sigy=sig;
sigx=r*sig;
R=[cos(phi) -sin(phi); sin(phi) cos(phi)];
C=R*diag([sigx,sigy])*R';

X=[x(:) y(:)];

Gb=gaussian(X,[0 0]',C);
Gb=reshape(Gb,N,N);

m=R*[0 sig]';
Ga=gaussian(X,m,C);
Ga=reshape(Ga,N,N);
Gc=rot90(Ga,2);

a=-1;
b=2;
c=-1;

G = a*Ga + b*Gb + c*Gc;

