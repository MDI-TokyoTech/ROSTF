function result = St(I,window,loc,use_GPU)
    %ST この関数の概要をここに記述
    %   詳細説明をここに記述
%     [rows, cols, chan] = size(I);
%     result = gpuArray(zeros([rows*window, cols*window, chan]));
%     result(1:window:rows*window, 1:window:cols*window, :) = I;
    [rows, cols, chan] = size(I);
%     result = gpuArray(zeros([rows*window, cols*window, chan]));
%     result(1:window:rows*window, 1:window:cols*window, :) = I;
    if use_GPU == 1
        mask = gpuArray(zeros(window, window));
    else
        mask = zeros(window, window);
    end
    if loc == 'lt'
        mask(1, 1) = 1;
    elseif loc=='c'
        mask(floor(window/2), floor(window/2)) = 1;
    end
   
    result = reshape(I, [rows, cols*chan]);
    result = kron(result, mask);
    result = reshape(result, rows*window, cols*window, chan);
end

