function result = Db(z)
    n3 = size(z, 3);
    if n3 == 1
        result = z;
    else
        result = z(:, :, [2:n3, n3]) - z;
    end
end