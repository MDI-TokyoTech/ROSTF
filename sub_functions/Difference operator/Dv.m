function result = Dv(z)
    n1 = size(z, 1);
    result = z([2:n1, n1], :, :) - z;
end