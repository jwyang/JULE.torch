
function [IM_result]=display_clust_image(IM,input_mask,fig)
% [IMseg]=display_clust_image(IM,input_mask,fig)
% display image segmentation results:
% input :
%          IM = input image
%          input_mask = segmentation mask
%          fig = matlab figure number for display
%
%  output:
%          IMseg = image with segments marked by different colors
%
%


if( nargin < 3 )
   fig = 0;
end

[height,width,colors] = size(IM);
segNum = max(max(input_mask));

if( size(IM,3) == 1 )
    IM = cat(3,IM,IM,IM);
end

map = [0 0 255; 150 0 255; 255 0 255; 255 0 0; 0 255 255; 0 255 87; 255 255 0; ...
        255 140 0; 128 0 0; 0 0 140; 255 196 125; 170 107 68]/255;
if( size(map,1)<segNum )
    map = hsv(segNum);
end
map_yiq = rgb2ntsc(map);

YIQ = rgb2ntsc(IM);
Y = YIQ(:,:,1);
I = YIQ(:,:,2);
Q = YIQ(:,:,3);
IM_result = ntsc2rgb(cat(3,Y,I,Q));


for i=1:segNum,
    mask = zeros(height,width);
    ind = find(input_mask==i);
    mask(ind) = 1;
    mask = medfilt2(mask);
    mask = medfilt2(mask);
    mask = medfilt2(mask);
    ind = find(mask);
    %%% mark segment boundary in white
    E = edge(mask);
    se = strel('disk',2);
    E = imdilate(E,se,'same');
    ind = find(E);
    %%% white
    Y(ind) = 1;
    I(ind) = 0;
    Q(ind) = 0;
    %%% green
    Y(ind) = 0.587;
    I(ind) = -0.2744;
    Q(ind) = -0.5299;   
    %%% red
    Y(ind) = 0.2989;
    I(ind) = 0.5959;
    Q(ind) = 0.2115;
end
IM_result = ntsc2rgb(cat(3,Y,I,Q));
if( fig>0 )
    figure(fig);
    clf;
    imshow(IM_result);
end



 
 
 