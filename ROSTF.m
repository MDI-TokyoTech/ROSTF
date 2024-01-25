function [preHt, optproc] = ROSTF(observed, groundtruth, params, output_dir)
    %% Observed images
    Hr = observed.Hr;
    Lr = observed.Lr;
    Lt = observed.Lt;
    chans = size(Hr,3);

    %% Ground-truth
    Hr_GT = groundtruth.Hr_GT;
    Ht_GT = groundtruth.Ht_GT;

    %% Initializing variables
    if params.use_GPU == 1
        preHr = gpuArray(ones(size(Hr))); 
        preHt = gpuArray(ones(size(Hr))); 
        preShr = gpuArray(ones(size(Hr))); 
        preSlr = gpuArray(ones(size(Lr)));
        preSlt = gpuArray(ones(size(Lt)));
        z1 = gpuArray(ones(size(D(Hr))));
        z2 = gpuArray(ones(size(D(Hr))));
        z3 = gpuArray(ones(size(D(Hr))));
        z4 = gpuArray(ones(size(Hr)));
        z5 = gpuArray(ones(size(Lr)));
        z6 = gpuArray(ones(size(Lt)));
    else
        preHr = ones(size(Hr)); 
        preHt = ones(size(Hr)); 
        preShr = ones(size(Hr)); 
        preSlr = ones(size(Lr));
        preSlt = ones(size(Lt));
        z1 = ones(size(D(Hr)));
        z2 = ones(size(D(Hr)));
        z3 = ones(size(D(Hr)));
        z4 = ones(size(Hr));
        z5 = ones(size(Lr));
        z6 = ones(size(Lt));
    end
   
    %% Setting functions
    B = @(z) UB(z,params.hsize,params.use_GPU);
    Bt = @(z) UBt(z,params.hsize,params.use_GPU);
    SB = @(z) S(B(z),params.window,params.downsampleloc);
    BtSt = @(z) Bt(St(z,params.window,params.downsampleloc,params.use_GPU));
    if params.p == 1
        projLpBall = @(z) projFastL1Ball(z, params.alpha);
    elseif params.p == 2
        projLpBall = @(z) proj_Fball(z, zeros(size(z)), params.alpha);
    end

    %% Making lists for recording optimization process
    objfuncval_list = zeros(1, params.max_iteration);
    psnr_Hr_list = zeros(1,params.max_iteration);
    psnr_Ht_list = zeros(1,params.max_iteration);
    error_preHr_list = zeros(1,params.max_iteration);
    error_preHt_list = zeros(1,params.max_iteration);
    rr_list = zeros(1, params.max_iteration);
    rt_list = zeros(1, params.max_iteration);
    mean_dif_list = zeros(chans, params.max_iteration);
    ite_time_list = zeros(1, params.max_iteration);
    preShr_l1_list = zeros(1, params.max_iteration);
    preSlr_l1_list = zeros(1, params.max_iteration);
    preSlt_l1_list = zeros(1, params.max_iteration);

    %% Algorithm starts.
    is_converged = 0;
    iteration = 1;
    band_for_show = [4 3 2];

    while is_converged == 0
    
        tic;
    
        % Updating primal variables
        preHr_next = preHr - params.gamma1(1) * (Dt(z1) + Dt(z3) + BtSt(z5) + z4); % steps 4,6
        preHt_next = preHt - params.gamma1(2) * (Dt(z2) - Dt(z3) + BtSt(z6)); % steps 5,7
        preShr_next = preShr - params.gamma1(3) * z4; % step 11
        preSlr_next = preSlr - params.gamma1(4) * z5; % step 12
        preSlt_next = preSlt - params.gamma1(5) * z6; % step 13
    
        for chan = 1:chans
            preHt_next(:,:,chan) = indicator_hyperslab(preHt_next(:,:,chan), ...
                                                       params.low(1,1,chan), ...
                                                       params.high(1,1,chan));  % step 9
        end
        preShr_next = projFastL1Ball(preShr_next, params.etah); % step 11
        preSlr_next = projFastL1Ball(preSlr_next, params.etal); % step 12
        preSlt_next = projFastL1Ball(preSlt_next, params.etal); % step 13
    
        % Updating dual variables
        Vr = 2*preHr_next - preHr; % step 14
        Vt = 2*preHt_next - preHt; % step 15
        Whr = 2*preShr_next - preShr; % step 16
        Wlr = 2*preSlr_next - preSlr; % step 17
        Wlt = 2*preSlt_next - preSlt; % step 18
        
        z1 = z1 + params.gamma2(1)*D(Vr); % step 19
        z2 = z2 + params.gamma2(2)*D(Vt); % step 20
        z3 = z3 + params.gamma2(3)*(D(Vr)-D(Vt)); % step 21
        z4 = z4 + params.gamma2(4)*(Vr + Whr); % step 22
        z5 = z5 + params.gamma2(5)*(SB(Vr) + Wlr); % step 23
        z6 = z6 + params.gamma2(6)*(SB(Vt) + Wlt); % step 24
        
        z1_next = z1 - params.gamma2(1)*prox12band(z1/params.gamma2(1), 1/params.gamma2(1)); % step 25
        z2_next = z2 - params.gamma2(2)*params.lambda*prox12band(z2/(params.gamma2(2)*params.lambda), 1/(params.gamma2(2)*params.lambda)); % step 26
        z3_next = z3 - params.gamma2(3)*projLpBall(z3/params.gamma2(3)); % step 27
        z4_next = z4 - params.gamma2(4)*proj_Fball(z4/params.gamma2(4), Hr, params.epsilonh); % step 28
        z5_next = z5 - params.gamma2(5)*proj_Fball(z5/params.gamma2(5), Lr, params.epsilonl); % step 29
        z6_next = z6 - params.gamma2(6)*proj_Fball(z6/params.gamma2(6), Lt, params.epsilonl); % step 30
        
        
        ite_time = toc;
        ite_time_list(iteration) = ite_time;
    
        % stopping condition
        error_preHr = (sqrt(sum((preHr - preHr_next).^2, "all")/sum(preHr.^2, "all")));
        error_preHt = (sqrt(sum((preHt - preHt_next).^2, "all")/sum(preHt.^2, "all")));
        rr = sqrt(sum((Lr - SB(preHr)).^2,'all'));
        rt = sqrt(sum((Lt - SB(preHt)).^2,'all'));
    
        if ((error_preHr <= params.stopping_criterion && ...
             error_preHt<= params.stopping_criterion && ...
             rr <= params.epsilonl && ...
             rt <= params.epsilonl) || ...
             iteration >= params.max_iteration)
            is_converged = 1;
        end
        
        % Preparing for the next iteration
        iteration = iteration + 1;
        preHr = preHr_next;
        preHt = preHt_next;
        preShr = preShr_next;
        preSlr = preSlr_next;
        preSlt = preSlt_next;
        z1 = z1_next;
        z2 = z2_next;
        z3 = z3_next;
        z4 = z4_next;
        z5 = z5_next;
        z6 = z6_next;
        
        % The value of the objective function at this iteration 
        objfuncval = HTV(preHr) + params.lambda*HTV(preHt);

        % The PSNR values of preHr and preHt 
        psnr_preHr = psnr(gather(preHr),Hr_GT);
        psnr_preHt = psnr(gather(preHt),Ht_GT);

        % The degree of satisfaction of the 2nd constraint
        mean_dif = zeros(chans);
        for chan = 1:chans
            mean_dif(chan) = params.c(chan) - mean(preHt(:,:,chan),'all');
        end

        % The degree of satisfaction of the 6-8th constraint
        preShr_l1 = l1norm(preShr);
        preSlr_l1 = l1norm(preSlr);
        preSlt_l1 = l1norm(preSlt);
        
        % Saving the above records
        objfuncval_list(iteration) = objfuncval;
        psnr_Hr_list(iteration) = psnr_preHr;
        psnr_Ht_list(iteration) = psnr_preHt;
        error_preHr_list(iteration) = error_preHr;
        error_preHt_list(iteration) = error_preHt;
        preShr_l1_list(iteration) = preShr_l1;
        preSlr_l1_list(iteration) = preSlr_l1;
        preSlt_l1_list(iteration) = preSlt_l1;
        rr_list(iteration) = rr;
        rt_list(iteration) = rt;
        for chan = 1:chans
            mean_dif_list(chan,iteration) = mean_dif(chan);
        end

        % Visualizing the current results once in 100 iterations
        if mod(iteration,100) == 0
            
            disp("iteration : " + num2str(iteration) + ...
                ", objval : " + num2str(objfuncval) + ...
                ", psnr of preHr : " + num2str(psnr_preHr) + ...
                ", psnr of preHt : " + num2str(psnr_preHt) + ...
                ", iteration time : "+ num2str(sum(ite_time_list)))
            
            figure(1);
            SBHr = SB(preHr);
            SBHt = SB(preHt);
            
            stretchr = Hr_GT(:,:,band_for_show);
            stretcht = Ht_GT(:,:,band_for_show);
            subplot(2,4,1), imshow(imadjust(Hr(:,:,band_for_show),stretchlim(stretchr),[])), title('$$ \mathbf{h_r} $$','Interpreter','latex','FontSize',20)
            subplot(2,4,2), imshow(imadjust(Lr(:,:,band_for_show),stretchlim(stretchr),[])), title('$$ \mathbf{l_r} $$','Interpreter','latex','FontSize',20)
            subplot(2,4,3), imshow(imadjust(Ht_GT(:,:,band_for_show),stretchlim(stretcht),[])), title('$$ (\mathbf{\hat{h}_t}) $$','Interpreter','latex','FontSize',20)
            subplot(2,4,4), imshow(imadjust(Lt(:,:,band_for_show),stretchlim(stretcht),[])), title('$$ \mathbf{l_t} $$','Interpreter','latex','FontSize',20)
            subplot(2,4,5), imshow(imadjust(preHr(:,:,band_for_show),stretchlim(stretchr),[])), title('$$ \mathbf{\tilde{h}_r} $$','Interpreter','latex','FontSize',20)
            subplot(2,4,6), imshow(imadjust(SBHr(:,:,band_for_show),stretchlim(stretchr),[])), title('$$ \mathbf{SB\tilde{h}_r} $$','Interpreter','latex','FontSize',20)
            subplot(2,4,7), imshow(imadjust(preHt(:,:,band_for_show),stretchlim(stretcht),[])), title('$$ \mathbf{\tilde{h}_t} $$','Interpreter','latex','FontSize',20)
            subplot(2,4,8), imshow(imadjust(SBHt(:,:,band_for_show),stretchlim(stretcht),[])), title('$$ \mathbf{SB\tilde{h}_t} $$','Interpreter','latex','FontSize',20)
            saveas(gcf,append(output_dir,'/current_result.png'))
            
            figure(2);
            plot(objfuncval_list);
            saveas(gcf,append(output_dir,'/current_objfuncval.png'))
            figure(3);
            plot(psnr_Ht_list);
            saveas(gcf,append(output_dir,'/current_PSNR.png'))
            
        end
    end
    
    %% Predicted HR image on the target date
    preHt = gather(preHt);
    
    %% Optimization Process
    optproc.objfuncval_list = objfuncval_list;
    optproc.psnr_Hr_list = psnr_Hr_list;
    optproc.psnr_Ht_list = psnr_Ht_list ;
    optproc.error_preHr_list = error_preHr_list;
    optproc.error_preHt_list = error_preHt_list;
    optproc.rr_list = rr_list;
    optproc.rt_list = rt_list;
    optproc.mean_dif_list = mean_dif_list;
    optproc.ite_time_list = ite_time_list;
    optproc.preShr_l1_list = preShr_l1_list;
    optproc.preSlr_l1_list = preSlr_l1_list;
    optproc.preSlt_l1_list = preSlt_l1_list;
end

