function [ key ] = generate_checksum( X, k ) 
    sum_ = sum(X(:));
    key = [num2str(sum_) '_' num2str(k)];
end

