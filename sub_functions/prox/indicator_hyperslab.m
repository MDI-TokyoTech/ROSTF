function result = indicator_hyperslab(z,low,high)
    
    % z^T 1
    zt1 = sum(z, 'all');
    n = numel(z);
    if zt1 <= low
        result = z + ((low - zt1)/n) * ones(size(zt1));
    elseif zt1 > high
        result = z + ((high - zt1)/n) * ones(size(zt1));
    else
        result = z;
    end

end

