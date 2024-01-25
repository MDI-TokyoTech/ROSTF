% 小野先生の記事から参照
% l1ノルムの近接写像
function result = prox_l1(A, gamma)
    result = sign(A).*max(abs(A) - gamma, 0);
end
