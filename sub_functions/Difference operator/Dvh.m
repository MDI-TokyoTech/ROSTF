function result = Dvh(z)
    n1 = size(z, 1);
    n2 = size(z, 2);
    result = cat(4, z([2:n1, n1], :, :) - z, z(:, [2:n2, n2], :) - z);
end