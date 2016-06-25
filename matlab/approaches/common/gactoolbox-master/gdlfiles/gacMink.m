function varargout = gacMink (X, k, dim)

if nargin < 3
    dim = 1;
end
switch nargout
    case 1
        sortedDist = gacPartial_sort(X, k, dim);
        varargout{1} = sortedDist;
    case 2
        [sortedDist, NNIndex] = gacPartial_sort(X, k, dim);
        varargout{1} = sortedDist;
        varargout{2} = NNIndex;
    otherwise
        error('too many output');
end

% check the first k elements
tmpSortedDist = sort(sortedDist, dim);
switch dim
    case {1, 2}
        if any(sortedDist(:) ~= tmpSortedDist(:))
            error('gacMink: the first k is unsorted!');
        end
    otherwise
        error('gacMink: only for matrices!');
end

end