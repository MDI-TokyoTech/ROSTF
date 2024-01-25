function result = UB(I,hsize,use_GPU)
    [rows, cols, chan] = size(I);

    % ガウスカーネルによるブラーリング
    psf = fspecial('average',hsize);
    psfsize = size(psf);
    blu = zeros(rows, cols);
    blu(1:psfsize(1), 1:psfsize(2)) = psf;
    if use_GPU == 1
        blu = gpuArray(circshift(blu, [-ceil((psfsize-1)/2) -ceil((psfsize-1)/2)]));
    else
        blu = circshift(blu, [-ceil((psfsize-1)/2) -ceil((psfsize-1)/2)]);
    end
    bluf = fft2(blu);
    bluf = repmat(bluf, [1,1,chan]);
    result = real(ifft2((fft2(I)).*bluf));    
end