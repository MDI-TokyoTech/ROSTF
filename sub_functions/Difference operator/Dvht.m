% function result = Dvht(z)
%     n1 = size(z, 1) / 2;
%     n2 = size(z, 2);
% 
%     z_v = z(1:n1, :, :);
%     z_h = z(n1+1:end, :, :);
% 
%     result_v = cat(1, -z_v(1, :, :), -z_v(2:(n1-1), :, :) + z_v(1:(n1-2), :, :), z_v(n1-1, :, :));
%     result_h = cat(2, -z_h(:, 1, :), -z_h(:, 2:(n2-1), :) + z_h(:, 1:(n2-2), :), z_h(:, n2-1, :));
% 
%     result = result_v + result_h;
% end

function result = Dvht(z)
    n1 = size(z, 1);
    n2 = size(z, 2);
    result = cat(1, -z(1, :, :, 1), -z(2:n1-1, :, :, 1) + z(1:n1-2, :, :, 1), z(n1-1, :, :, 1)) ...
        + cat(2, -z(:, 1, :, 2), -z(:, 2:n2-1, :, 2) + z(:, 1:n2-2, :, 2), z(:, n2-1, :, 2));
end