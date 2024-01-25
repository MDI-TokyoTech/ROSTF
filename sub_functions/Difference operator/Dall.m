function result = Dall(z)
    n1 = size(z, 1);
    n2 = size(z, 2);
    n3 = size(z, 3);
    result = cat(4, z([2:n1, n1], :, :) - z, z(:, [2:n2, n2], :) - z, z(:, :, [2:n3, n3]) - z);
end