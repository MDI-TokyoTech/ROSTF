function [p,r,ss,cc,sm] = cal_metrics(pre, gt)
        
        % psnr
        p = psnr(pre, gt);
        
        % rmse
        r = sqrt(immse(pre, gt));
        
        % ssim
        ss = ssim(pre, gt);

        % corrcoef
        cc = corrcoef(pre, gt); cc = cc(1,2);
        
        % sam
        sm = mean(acos(sum(pre.*gt,3)./sqrt(sum(pre.^2,3).*sum(gt.^2,3))),'all','omitnan');
end

