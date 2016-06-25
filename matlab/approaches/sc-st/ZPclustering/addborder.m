%%%%%
function imbig = addborder(im,xbdr,ybdr,arg);
% imnew = addborder(im,xborder,yborder,arg) Make image w/added border.
% imnew = addborder(im,5,5,128) Add 5 wide border of val 128.
% imnew = addborder (im,5,5,'even')             Even reflection.
% imnew = addborder (im,5,5,'odd')              Odd reflection.
% imnew = addborder (im,5,5,'wrap')             Wraparound.

[ysize, xsize] = size(im);

%--------if arg is a number, fill border with its value.
if ~isstr(arg)
        imbig = arg*ones((ysize + 2*ybdr), (xsize + 2*xbdr));
        imbig((ybdr+1):(ybdr+ysize), (xbdr+1):(xbdr+xsize)) = im;
        
%--------Else use a reflection method. First check thickness.
elseif ((xbdr > xsize-1) | (ybdr > ysize-1)),
         disp('borders must be thinner than image');

% -----------------------Even reflections.
elseif strcmp(arg,'even')
        %--set up array
        imbig = zeros((ysize + 2*ybdr), (xsize + 2*xbdr));
        imbig((ybdr+1):(ybdr+ysize), (xbdr+1):(xbdr+xsize)) = im;
        %--do reflections
        if (xbdr >= 1),
                imbig((ybdr+1):(ybdr+ysize), xbdr:-1:1) =...
                        im(:,1:xbdr);
                imbig((ybdr+1):(ybdr+ysize),(xbdr+xsize+1):(2*xbdr+xsize))...
                                 = im(:,xsize:-1:xsize-xbdr+1);
        end
        if (ybdr >= 1),
                imbig(ybdr:-1:1,:) = imbig(ybdr+1:ybdr+ybdr,:);
                imbig((ybdr+ysize+1):(2*ybdr+ysize), :)...
                         = imbig(ybdr+ysize:-1:ysize+1, :);
        end

% ---------------------odd reflections.
elseif strcmp(arg,'odd')
%--set up array
        imbig = zeros((ysize + 2*ybdr), (xsize + 2*xbdr));
        imbig((ybdr+1):(ybdr+ysize), (xbdr+1):(xbdr+xsize)) = im;
%--do reflections.  Skip it if bdr is 0 or less.
        if (xbdr >= 1),
                imbig((ybdr+1):(ybdr+ysize), xbdr:-1:1) = im(:,2:xbdr+1);
                imbig((ybdr+1):(ybdr+ysize),(xbdr+xsize+1):(2*xbdr+xsize))...
                         = im(:,xsize-1:-1:xsize-xbdr);
        end
        if (ybdr >= 1),
                imbig(ybdr:-1:1,:) = imbig(ybdr+2:ybdr+ybdr+1,:);
                imbig((ybdr+ysize+1):(2*ybdr+ysize), :)...
                         = imbig(ybdr+ysize-1:-1:ysize, :);
        end
                 
% ---------------------Wraparound
elseif strcmp(arg,'wrap')
%--set up array
        imbig = zeros((ysize + 2*ybdr), (xsize + 2*xbdr));
        imbig((ybdr+1):(ybdr+ysize), (xbdr+1):(xbdr+xsize)) = im;
%--do reflections
        if (xbdr >= 1),
                imbig((ybdr+1):(ybdr+ysize), 1:xbdr) = im(:,xsize-xbdr+1:xsize);
                imbig((ybdr+1):(ybdr+ysize),(xbdr+xsize+1):(2*xbdr+xsize))...
                         = im(:,1:xbdr);
        end
        if (ybdr >= 1),
                imbig(1:ybdr,:) = imbig(ysize+1:ysize+ybdr,:);
                imbig((ybdr+ysize+1):(2*ybdr+ysize), :)...
                         = imbig(ybdr+1:ybdr+ybdr, :);
        end
        
else  error('unknown border style');

end