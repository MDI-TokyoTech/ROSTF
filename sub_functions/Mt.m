function result = Mt(I,window)
    %Mt この関数の概要をここに記述
    n1 = size(I,1);
    n2 = size(I,2);
    fil = gpuArray(ones(window, window));
    Iconv = convn(I, fil, 'full');
    result = Iconv(window:window:n1+window-1, window:window:n2+window-1, :);
end