
function [D,ind_non_zero,rows_nonz,cols_nonz] = imdist(IM,R)
%
% [D,ind_non_zero,rows_nonz,cols_nonz] = imdist(IM,R)
% build a sparse distance matrix for image IM connecting pixels within
% radius R
%
% Lihi Zelnik-Manor, March 2005, Caltech
%

[rows,cols,colors] = size(IM);
[x,y] = meshgrid(1:rows,1:cols);
ndata = rows*cols;

%% make sure image values are within 0-1 and not 0-255
if( max(IM(:)) > 1 )
    IM = IM/255;
end


%% build distance matrix for first color channel
[D,rows_nonz,cols_nonz] = mkImDist(IM(:,:,1),R);
ind_non_zero = sub2ind(size(D),rows_nonz,cols_nonz);
    
%% if there is more than one color channel sum all the distances
% note, that all channels have the same fill pattern in the distance
% matrix
for c = 2:colors,
    D1 = mkImDist(IM(:,:,c),R);
    D = D+D1;
    clear D1; 
end

return;


%%
function [D,ind2,ind1]=mkImDist(im,R)
%function [D,row_ind,col_ind]=mkDistmatrix(im,R)
% use block processing to make a distance matrix
% D is a matrix size numberofpixels x numberofpixels
% D(i,j)=(im(i)-im(j)).^2
% where i and j are pixel locations
% R defines the neighborhood size

[sx,sy]=size(im);
IndImage=reshape(1:(sx*sy),sx,sy);
imBig=addborder(im,(R-1)/2,(R-1)/2,0);
IndImage=addborder(IndImage,(R-1)/2,(R-1)/2,0);

A=im2col(imBig,[R R],'sliding');
Ind=im2col(IndImage,[R R],'sliding');
% now each colunn of A is a neighborhood of im
% we want to subtract the values of the center row

[xA,yA]=size(A);
centerPixel=(xA+1)/2;
O=ones(xA,1);
A2=O*A(centerPixel,:);
Ind2=O*Ind(centerPixel,:);
I=find(Ind(:)>0 & Ind2(:)>0);

ind1 = Ind(I);    clear Ind;
ind2 = Ind2(I);    clear Ind2;
a = A(I);   clear A;
a2 = A2(I);   clear A2;

Dff=(a-a2).^2 + eps;   %% avoid true zeros, put eps instead
D=sparse(ind2,ind1,Dff);

return;
