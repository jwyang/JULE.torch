% function [x,y,gx,gy,par,threshold,mag,mage,g,FIe,FIo,mago] = quadedgep(I,par,threshold);
% Input:
%    I = image
%    par = vector for 4 parameters
%      [number of filter orientations, number of scales, filter size, elongation]
%      To use default values, put 0.
%    threshold = threshold on edge strength
% Output:
%    [x,y,gx,gy] = locations and gradients of an ordered list of edgels
%       x,y could be horizontal or vertical or 45 between pixel sites
%       but it is guaranteed that there [floor(y) + (floor(x)-1)*nr] 
%       is ordered and unique.  In other words, each edgel has a unique pixel id.
%    par = actual par used
%    threshold = actual threshold used
%    mag = edge magnitude
%    mage = phase map
%    g = gradient map at each pixel
%    [FIe,FIo] = odd and even filter outputs
%    mago = odd filter output of optimum orientation
%
% Stella X. Yu, 2001



function [x,y,gx,gy,par,threshold,mag,mage,g,FIe,FIo,mago] = quadedgep(I,par,threshold);

if nargin<3 | isempty(threshold),
    threshold = 0.2;
end

[r,c] = size(I);
def_par = [8,1,20,3];

% take care of parameters, any missing value is substituted by a default value
if nargin<2 | isempty(par),
   par = def_par;
end
par(end+1:4)=0;
par = par(:);
j = (par>0);
have_value = [ j, 1-j ];
j = 1; n_filter = have_value(j,:) * [par(j); def_par(j)];
j = 2; n_scale  = have_value(j,:) * [par(j); def_par(j)];
j = 3; winsz    = have_value(j,:) * [par(j); def_par(j)];
j = 4; enlong   = have_value(j,:) * [par(j); def_par(j)];

% always make filter size an odd number so that the results will not be skewed
j = winsz/2;
if not(j > fix(j) + 0.1),
    winsz = winsz + 1;
end

% filter the image with quadrature filters
FBo = make_filterbank_odd2(n_filter,n_scale,winsz,enlong);
FBe = make_filterbank_even2(n_filter,n_scale,winsz,enlong);
n = ceil(winsz/2);
f = [fliplr(I(:,2:n+1)), I, fliplr(I(:,c-n:c-1))];
f = [flipud(f(2:n+1,:)); f; flipud(f(r-n:r-1,:))];
FIo = fft_filt_2(f,FBo,1); 
FIo = FIo(n+[1:r],n+[1:c],:);
FIe = fft_filt_2(f,FBe,1);
FIe = FIe(n+[1:r],n+[1:c],:);

% compute the orientation energy and recover a smooth edge map
% pick up the maximum energy across scale and orientation
% even filter's output: as it is the second derivative, zero cross localize the edge
% odd filter's output: orientation
mag = sqrt(sum(FIo.^2,3)+sum(FIe.^2,3));
mag_a = sqrt(FIo.^2+FIe.^2);
[tmp,max_id] = max(mag_a,[],3);
base_size = r * c;
id = [1:base_size]';
mage = reshape(FIe(id+(max_id(:)-1)*base_size),[r,c]);
mage = (mage>0) - (mage<0);

ori_incr=pi/n_filter; % to convert jshi's coords to conventional image xy
ori_offset=ori_incr/2;
theta = ori_offset+([1:n_filter]-1)*ori_incr; % orientation detectors
% [gx,gy] are image gradient in image xy coords, winner take all
mago = reshape(FIo(id+(max_id(:)-1)*base_size),[r,c]);
ori = theta(max_id);
ori = ori .* (mago>0) + (ori + pi).*(mago<0);
gy = mag .* cos(ori);
gx = -mag .* sin(ori);
g = cat(3,gx,gy);

% phase map: edges are where the phase changes
mag_th = max(mag(:)) * threshold;
eg = (mag>mag_th);
h = eg & [(mage(2:r,:) ~= mage(1:r-1,:)); zeros(1,c)];
v = eg & [(mage(:,2:c) ~= mage(:,1:c-1)), zeros(r,1)];
[y,x] = find(h | v);
k = y + (x-1) * r;
h = h(k);
v = v(k);
y = y + h * 0.5; % i
x = x + v * 0.5; % j
t = h + v * r;
gx = g(k) + g(k+t);
k = k + (r * c);
gy = g(k) + g(k+t);


