function result = UBt(I,hsize,use_GPU)

    [rows, cols, chan] = size(I);

    % ガウスカーネルによるブラーリングの転置
    psf = fspecial('average',hsize);
    psfsize = size(psf);
    blu = zeros(rows, cols);
    blu(1:psfsize(1), 1:psfsize(2)) = psf;

    if use_GPU == 0
        blu = circshift(blu, [-ceil((psfsize-1)/2) -ceil((psfsize-1)/2)]);
    else
        blu = gpuArray(circshift(blu, [-ceil((psfsize-1)/2) -ceil((psfsize-1)/2)]));
    end
%     if use_GPU == 1
%         blu = gpuArray(circshift(blu, [-ceil((psfsize-1)/2) -ceil((psfsize-1)/2)]));
%     else
%         blu = circshift(blu, [-ceil((psfsize-1)/2) -ceil((psfsize-1)/2)]);
%     end
    bluf = fft2(blu);
    bluft = conj(bluf);
    bluft = repmat(bluft, [1,1,chan]);
    result = real(ifft2((fft2(I)).*bluft));
end