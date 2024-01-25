function result = Dbt(z)
    n3 = size(z, 3);
    if n3 == 1
        result = z;
    else
        result = cat(3, -z(:, :, 1), -z(:, :, 2:n3-1) + z(:, :, 1:n3-2), z(:, :, n3-1));
    end
end