clear
close all

kernel_options = {...
    'gaussian_symmetric', ...
    'gaussian_without_tilt', ...
    'gaussian_tilt', ...
    'non_gaussian'};

[idx, tf] = listdlg('PromptString','Select blur kernel type:', ...
                    'SelectionMode','single', ...
                    'ListString',kernel_options);

if ~tf
    error('No kernel type selected. Program terminated.');
end

kernel_type = kernel_options{idx};

cof=1.55;

switch kernel_type
    case 'gaussian_symmetric'
        data_path = '../dataset/gaussian kernels with symmetric/';
        nkt = 13;
        nk = 13;
        
    case 'gaussian_without_tilt'
        data_path = '../dataset/gaussian kernels without tilt/';
        nkt = 13;
        nk = 13;

    case 'gaussian_tilt'
        data_path = '../dataset/gaussian kernels with tilt/';
        nkt = 13;
        nk = 13;

    case 'non_gaussian'
        data_path = '../dataset/non gaussian kernels/';
        nkt = 54;
        nk = 54;

    otherwise
        error('Unknown kernel type selected.');
end

ns=100; %number of sharp images for train 
ngtt=50; %number of sharp images for test 

nb=nk*ns; %number of blur images for train %13*100 %54*100
nt=nkt*ngtt; %number of blur images for test %13*50 %54*50

load(fullfile(data_path, 'size_k.mat'));
load(fullfile(data_path, 'em.mat'));
load(fullfile(data_path, 'blur_kernels.mat'));

if strcmp(kernel_type, 'non_gaussian')
    sk = size_k';
    size_k = (max(sk))';
    em = size_k / (2 * cof);
end

emt=em;

sig_tr=repmat(em,ns,1);
rep_sk_tr=repmat(size_k,ns,1);

sig_t=repmat(emt,ngtt,1);
rep_sk_t=repmat(size_k,ngtt,1);

%feature extraction from training images:
tic;
[r_s,r_b]=RUN_HOG_B_S(ns,nb,nk,kernel_type); 
t1=toc;
save('r_b.mat','r_b','-v7.3');
save('r_s.mat','r_s','-v7.3');

%feature extraction from test images:
tic;
[r_t,r_gtt]=RUN_HOG_B_test(nt,ngtt,nkt,kernel_type); 
t2=toc;
save('r_t.mat','r_t','-v7.3');
save('r_gtt.mat','r_gtt','-v7.3');

%train models:
tic;
[mdl]=RUN_SVR_Train(nb,r_s,r_b,sig_tr,rep_sk_tr);
t3=toc;
save('mdl.mat','mdl','-v7.3');

%test models:
tic;
[TRsig,sigma,xj,yj,fi]=RUN_SVR_Predict(nt,r_t,r_gtt,mdl,sig_t,rep_sk_t);
t4=toc;
sigma_x = sigma(:,1);
sigma_y = sigma(:,2);
outputs=[sigma_x,sigma_y,sig_t,rep_sk_t];

time_of_feature_extraction_from_train_images=t1;
time_of_feature_extraction_from_test_images=t2;
time_of_train_models=t3;
time_of_test_models=t4;
times=[t1;t2;t3;t4];
total_processing_time_minute = sum(times)/60

max_sig = max(sigma_x,sigma_y);
kSizeOut = cof .* max_sig;

rks=fix(abs(kSizeOut));
rks1=rks;

for ii=1:nt
    if (mod(rks(ii,1),2)==0)
    rks1(ii,1)=rks(ii,1)+1;
    end
end

kernel_size_out=rks1;
test_size_blur_kernels=rep_sk_t;

pixel_dif = abs(kernel_size_out-test_size_blur_kernels);
error_percentage = (pixel_dif./test_size_blur_kernels)*100;
status = pixel_dif <= 20;
num_ones = sum(status);
correct_rate = (num_ones / length(pixel_dif))*100

data_table = table( test_size_blur_kernels,kernel_size_out,pixel_dif,error_percentage, 'VariableNames', {'gt_kernel_size','Kernel_size_Out','pixel_dif','error_percentage'});

csv_file_path = 'kernel_size_out.csv';

writetable(data_table, csv_file_path);
