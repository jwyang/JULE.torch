%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Normalized Cut Segmentation Code                              %
%						                                            %	
%  Timothee Cour (INRIA), Stella Yu (Berkeley), Jianbo Shi (UPENN)  %
%     						                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

License
This software is made publicly for research use only. It may be modified and redistributed under the terms of the GNU General Public License. 

Citation
Please cite the following if you plan to use the code in your own work: 
* Normalized Cuts and Image Segmentation, Jianbo Shi and Jitendra Malik, IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI) 2000
* Normalized Cut Segmentation Code, Timothee Cour, Stella Yu, Jianbo Shi. Copyright 2004 University of Pennsylvania, Computer and Information Science Department.

Tested on matlab R2009b.

Installation Notes :

1) After you unzipped the files to mydir, 
   put the Current Directory in Matlab to mydir

2) In the matlab command prompt,
   type compileDir_simple to compile the mex files (ignore the error on the C++ non-mex file; needs to be done once)

3) You can now try any of the functions

type demoNcutImage to see a demo of image segmentation
type demoNcutClustering to see a demo of point cloud clustering


Other top level functions:

NcutImage.m: given image "I", segment it into "nbSegments" segments
    [SegLabel,NcutDiscrete,NcutEigenvectors,NcutEigenvalues,W]= NcutImage(I,nbSegments);
    
ICgraph.m: compute Intervening Contour based pixel similarity matrix W
    W = ICgraph(I);
    
ncutW.m: Given a similarity graph "W", computes Ncut clustering on the graph into "nbSegments" groups;
    [NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(W,nbSegments);


Release notes:

2010, January 22: release of all c++ source mex files compatible with matlab R2009b
2006, May 04: release version 8: fixed incompatibility issues with new matlab
2004, June 18: release version 7: initial release

Maintained by Timothee Cour, timothee dot cour at gmail dot com

January 22, 2010.