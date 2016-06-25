function [data,size_cluster] = build_scene(caseid)
% [data,size_cluster] = build_scene(case)
%
% case 1: random gaussian shaped blobs.
% case 2: one circular plus some gaussian blob, one inside, one outside 
% case 3: one circular plus some gaussian blob, one inside, two outside 
% Jianbo Shi, 1997


if caseid ==1,
  sigma_h = 2;
  sigma_v = 10;

  s_v = 10;
  s_h = 30;

  a = [sigma_h*randn(1,40);sigma_v*randn(1,40)];
  b = [s_h;s_v]*ones(1,50) + [sigma_h*randn(1,50);...
        sigma_v*randn(1,50)];

  data = [a,b];
  size_cluster = [40,50];

elseif caseid == 2,
   num_cluster = 3;
   radius = 15;
   size_cluster = [80,20,20];

   raw_data = randn(2,sum(size_cluster));
   tmp = rand(2,size_cluster(1))-0.5;

   [t,idt] = sort(tmp(2,:));
   r_noise = 4;
   raw_data2 = [tmp(1,idt)*r_noise;...
             tmp(2,idt)*2];

   data = [(radius-raw_data2(1,1:size_cluster(1))).*...
        cos(pi*raw_data2(2,1:size_cluster(1)));...
        (radius-raw_data2(1,1:size_cluster(1))).*...
        sin(pi*raw_data2(2,1:size_cluster(1)))];

   
   center = [0,0];sig = [1,2];
   % size_cluster_base
   scb = size_cluster(1)+1;
   scb_next = scb+size_cluster(2)-1;
   data = [data,[center(1)+sig(1)*raw_data(1,scb:scb_next);...
                center(2)+sig(2)*raw_data(2,scb:scb_next)]];


   center = [radius+10,0]; sig = [1,1];
   scb = scb_next+1;
   scb_next = scb+size_cluster(3)-1;
   data = [data,[center(1)+sig(1)*raw_data(1,scb:scb_next);...
                center(2)+sig(2)*raw_data(2,scb:scb_next)]];
elseif caseid==3,
   num_cluster = 4;
   radius = 15;
   size_cluster = [80,10,20,20];

   raw_data = randn(2,sum(size_cluster));
   tmp = rand(2,size_cluster(1))-0.5;

   [t,idt] = sort(tmp(2,:));
   r_noise = 4;
   raw_data2 = [tmp(1,idt)*r_noise;...
             tmp(2,idt)*2];

   data = [(radius-raw_data2(1,1:size_cluster(1))).*...
        cos(pi*raw_data2(2,1:size_cluster(1)));...
        (radius-raw_data2(1,1:size_cluster(1))).*...
        sin(pi*raw_data2(2,1:size_cluster(1)))];

   
   center = [0,0];sig = [1,2];
   % size_cluster_base
   scb = size_cluster(1)+1;
   scb_next = scb+size_cluster(2)-1;
   data = [data,[center(1)+sig(1)*raw_data(1,scb:scb_next);...
                center(2)+sig(2)*raw_data(2,scb:scb_next)]];


   center = [radius+25,8]; sig = [1,2.3];
   scb = scb_next+1;
   scb_next = scb+size_cluster(3)-1;
   data = [data,[center(1)+sig(1)*raw_data(1,scb:scb_next);...
                center(2)+sig(2)*raw_data(2,scb:scb_next)]];

   center = [radius+25,-6]; sig = [1.5,2.4];
   scb = scb_next+1;
   scb_next = scb+size_cluster(4)-1;
   data = [data,[center(1)+sig(1)*raw_data(1,scb:scb_next);...
                center(2)+sig(2)*raw_data(2,scb:scb_next)]];
elseif caseid == 4,
   size_cluster = [100,10,10];
   radius = 10;
   tmp = rand(2,size_cluster(1))-0.5;

   [t,idt] = sort(tmp(2,:));
   r_noise = 4;
   raw_data2 = [tmp(1,idt)*r_noise;...
             tmp(2,idt)*2];

   data = [(radius-raw_data2(1,1:size_cluster(1))).*...
        cos(pi*raw_data2(2,1:size_cluster(1)));...
        (radius-raw_data2(1,1:size_cluster(1))).*...
        sin(pi*raw_data2(2,1:size_cluster(1)))];

   
   result = zeros(1,size_cluster(1));

  % for j =1:size_cluster(1),
%	result(j) = sum(sum(A(1:j,1:j)))/j;
%   end

end
