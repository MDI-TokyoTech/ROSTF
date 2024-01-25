function result = M(I,window,use_GPU)
    %M この関数の概要をここに記述
    [rows, cols, chan] = size(I);
    if use_GPU == 1
        mask = gpuArray(ones(window, window));
    else
        mask = ones(window, window);
    end
    result = reshape(I, [rows, cols*chan]);
    result = kron(result, mask);
    result = reshape(result, rows*window, cols*window, chan);
end