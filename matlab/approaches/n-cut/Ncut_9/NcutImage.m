function [SegLabel,NcutDiscrete,NcutEigenvectors,NcutEigenvalues,W,imageEdges]= NcutImage(I,nbSegments);
%  [SegLabel,NcutDiscrete,NcutEigenvectors,NcutEigenvalues,W,imageEdges]= NcutImage(I);
%  Input: I = brightness image
%         nbSegments = number of segmentation desired
%  Output: SegLable = label map of the segmented image
%          NcutDiscrete = Discretized Ncut vectors
%  
% Timothee Cour, Stella Yu, Jianbo Shi, 2004.


 
if nargin <2,
   nbSegments = 10;
end

[W,imageEdges] = ICgraph(I);

[NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(W,nbSegments);

%% generate segmentation label map
[nr,nc,nb] = size(I);

SegLabel = zeros(nr,nc);
for j=1:size(NcutDiscrete,2),
    SegLabel = SegLabel + j*reshape(NcutDiscrete(:,j),nr,nc);
end