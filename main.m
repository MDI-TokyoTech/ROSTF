%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the demo file of the method proposed in the
% following reference:
% 
% R.Isono, K. Naganuma, and S. Ono
% ``Robust Spatiotemporal Fusion of Satellite Images: A Constrained Convex Optimization Approach.''
%
% Update history:
% October 7, 2023: v1.0 
%
% Copyright (c) 2023 Ryosuke Isono, Kazuki Naganuma, and Shunsuke Ono
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Adding path
addpath(genpath('./sub_functions'));

for p = [1 2]
for site_idx = [1 2]
for datatype_idx = [1 2]
for nsigh = [0 0.05]
for sprateh = [0 0.05]

%% Loading data
% choosing data
if site_idx == 1
    site = 'Site1'; % 'Site1' or 'Site2'
elseif site_idx == 2
    site = 'Site2';
end
if datatype_idx == 1
    datatype = 'Sim';
elseif datatype_idx == 2
    datatype = 'Real';
end 

nsigl = 0; % standard deviation of Gaussian noise on LR images
spratel = 0; % superimposition ratio of sparse noise on LR images

% loading data
data_file = append('dataset/',site,'/',datatype,'/');
data_file = append(data_file,'nsigh=',num2str(nsigh));
data_file = append(data_file,'_nsigl=',num2str(nsigl));
data_file = append(data_file,'_sprateh=',num2str(sprateh));
data_file = append(data_file,'_spratel=',num2str(spratel));
data_file = append(data_file,'.mat');
load(data_file)
observed.Hr = Hr;
observed.Lr = Lr;
observed.Lt = Lt;
groundtruth.Hr_GT = Hr_GT;
groundtruth.Ht_GT = Ht_GT;
groundtruth.Lr_GT = Lr_GT;
groundtruth.Lt_GT = Lt_GT;

% getting the size of the images
nH = numel(Hr_GT);
nL = numel(Lr_GT);
[rowsH, colsH, chans] = size(Hr_GT);
[rowsL, colsL, chans] = size(Lr_GT);
nHc = rowsH*colsH; % Hの1chanにおける画素数
nLc = rowsL*colsL; % Lの1chanにおける画素数

%% Setting parameters
% use GPU or not
params.use_GPU = 0;  % 0 if you do not use GPU, 1 if you use GPU

% the way of downsampling
params.hsize = 20;  % size of averaging filter as a blurring operator
params.window = 20; % size of downsampling window
params.downsampleloc = 'c'; % location of the pixel taken when downsampling ('lt': lefttop or 'c': center)
B = @(z) UB(z,params.hsize,params.use_GPU);
Bt = @(z) UBt(z,params.hsize,params.use_GPU);
SB = @(z) S(B(z),params.window,params.downsampleloc);
BtSt = @(z) Bt(St(z,params.window,params.downsampleloc,params.use_GPU));

% the balancing parameter
params.lambda = 1;

% calculating edge similarity using the p-norm
params.p = p;

% the 1st constraint
if params.p == 1
    params.alpha_coef = 0.002; % recommend 0.002 when p=1
elseif params.p == 2
    params.alpha_coef = 10^(-6); % recommend 10^(-6) when p=2
end
params.alpha = params.alpha_coef * nH;

% the 2nd constraint
params.beta = zeros(1,1,chans);
params.c = zeros(1,1,chans); 
params.low = zeros(1,1,chans);
params.high = zeros(1,1,chans);
for chan = 1 : chans
    mLr = mean(Lr(:,:,chan),'all');
    mHr = mean(Hr(:,:,chan),'all');
    mLt = mean(Lt(:,:,chan),'all');

    params.beta(1,1,chan) = abs(mLr - mHr); 
    params.c(1,1,chan) = mLt - (0.5 - mLt) * spratel;

    params.low(1,1,chan) = (params.c(1,1,chan) - params.beta(1,1,chan))*nHc;
    params.high(1,1,chan) = (params.c(1,1,chan) + params.beta(1,1,chan))*nHc;
end

% the 3rd constraint
params.epsilonh_coef = 0.98; 
params.epsilonh = params.epsilonh_coef * nsigh * sqrt(nH*(1 - sprateh));

% the 4-5th constraint
params.epsilonl = sqrt(sum((Lr - SB(Hr)).^2,'all'));  % || Lr - SBHr ||_2 

% the 6th constraint
params.etah_coef = 0.98;
params.etah = params.etah_coef * sprateh * nH * 0.5;

% the 7th constraint
params.etal_coef = 0.98;
params.etal = params.etal_coef * spratel * nL * 0.5;

% the stopping criterion of Algoritm1
params.max_iteration = 50000;
params.stopping_criterion = 0.00001;

% the stepsizes of Algorithm1
params.gamma1 = [1/18 1/17 1 1 1];
params.gamma2 = [1/5 1/5 1/5 1/5 1/5 1/5];

%% Making output directory
output_dir = append('Results/ROSTF-',num2str(params.p));
output_dir = append(output_dir, '/', site, '_');
output_dir = append(output_dir, datatype);
output_dir = append(output_dir, '_nsigh=', num2str(nsigh));
output_dir = append(output_dir, '_nsigl=', num2str(nsigl));
output_dir = append(output_dir, '_sprateh=', num2str(sprateh));
output_dir = append(output_dir, '_spratel=', num2str(spratel));
mkdir(output_dir)
save(append(output_dir,'/config.mat'),'-struct','params')

%% Run ROSTF
[preHt, optproc] = ROSTF(observed, groundtruth, params, output_dir);

%% Evaluation
% psnr
psnr_preHt = psnr(preHt, Ht_GT);

% ssim
ssim_preHt = ssim(preHt, Ht_GT);

% rmse
rmse_preHt = sqrt(immse(preHt, Ht_GT));

%corrcoef
cc_preHt = corrcoef(preHt, Ht_GT); cc_preHt = cc_preHt(1,2);

disp('Evaluation')
disp(append('PSNR : ',num2str(psnr_preHt)))
disp(append('SSIM : ',num2str(ssim_preHt)))
disp(append('RMSE : ',num2str(rmse_preHt)))
disp(append('CC   : ',num2str(cc_preHt)))

%% Saving the resutls
save(append(output_dir,'/final_preHt.mat'),'preHt')
save(append(output_dir,'/optimization_process.mat'),'-struct','optproc')

end
end
end
end
end
