function result = Dht(z)
    n2 = size(z, 2);
    result = cat(2, -z(:, 1, :), -z(:, 2:(n2-1), :) + z(:, 1:(n2-2), :), z(:, n2-1, :));
end