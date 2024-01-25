x = [1 3 2; 2 4 5];
x = max(abs(x)-max(max((cumsum(sort(abs(x),1,'descend'),1)-alpha)./(1:size(x,1))'),0),0).*sign(x);