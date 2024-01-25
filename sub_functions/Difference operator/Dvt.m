function result = Dvt(z)
    n1 = size(z, 1);
    result = cat(1, -z(1, :, :), -z(2:(n1-1), :, :) + z(1:(n1-2), :, :), z(n1-1, :, :));
end