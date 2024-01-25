function result = HTV(X)
    Y1 = sum(X.^2,4);
    Y2 = sqrt(sum(Y1,3));
    result = sum(Y2,[1 2]);
end

