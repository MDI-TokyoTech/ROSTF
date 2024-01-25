function[X] = projFastL1Ball_withCenterV(X,V,alpha)

    % The trick used here is that we compute the threshold like above, and if it is negative, 
    % then y is inside the ball, so there is nothing to do and the threshold is set to zero.
    x = X(:);
    v = V(:); % Center
    y = x - v;
    y = max(abs(y)-max(max((cumsum(sort(abs(y),1,'descend'),1)-alpha)./(1:size(y,1))'),0),0).*sign(y);
    x = y + v;
    X = reshape(x,size(X));
end