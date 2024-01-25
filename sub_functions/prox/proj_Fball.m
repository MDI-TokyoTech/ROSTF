% this function is the proximal operation of
% \iota_{\|\cdot-V\|_{F}<=\epsilon}(X)
% input is a 3D tensor

function result = proj_Fball(X, V, epsilon)
%     l2distance = sqrt(sum(sum(sum((X - V).^2))));
    if ndims(X) == 3
        l2distance = sqrt(sum(sum(sum((X - V).^2))));
    end
    if ndims(X) == 4
        l2distance = sqrt(sum(sum(sum(sum((X - V).^2)))));
    end
     if l2distance <= epsilon
        result = X;
    else
        result = V + epsilon*(X - V)/l2distance;
    end
end