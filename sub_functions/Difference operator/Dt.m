function result = Dt(z)
    n1 = size(z, 1);
    n2 = size(z, 2);
    result = cat(1, -z(1, :, :, 1), -z(2:n1-1, :, :, 1) + z(1:n1-2, :, :, 1), z(n1-1, :, :, 1)) + cat(2, -z(:, 1, :, 2), -z(:, 2:n2-1, :, 2) + z(:, 1:n2-2, :, 2), z(:, n2-1, :, 2));
end