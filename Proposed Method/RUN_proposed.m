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

nb=nkt*100; %number of blur images for train
nt=nk*50; %number of blur images for test

load(fullfile(data_path, 'size_k.mat'));
load(fullfile(data_path, 'em.mat'));
load(fullfile(data_path, 'blur_kernels.mat'));

if strcmp(kernel_type, 'non_gaussian')
    sk = size_k';
    size_k = (max(sk))';
    em = size_k / (2 * cof);
end

train_sigma_gaussians=repmat(em,(nb/nkt),1);
train_size_blur_kernels=repmat(size_k,(nb/nkt),1);

test_sigma_gaussians=repmat(em,(nt/nk),1);
test_size_blur_kernels=repmat(size_k,(nt/nk),1);

%feature extraction from training images:
tic;
[r_s,r_b]=RUN_HOG_B_S_A(nb,nk,kernel_type); 
t1=toc;
save('r_b.mat','r_b','-v7.3');
save('r_s.mat','r_s','-v7.3');

%feature extraction from test images:
tic;
[r_t,r_gtt]=RUN_HOG_B_test_A(nt,nk,kernel_type); 
t2=toc;
save('r_t.mat','r_t','-v7.3');
save('r_gtt.mat','r_gtt','-v7.3');

%train models:
tic;
[mdl,mdls]=RUN_SVR_Train_A(nb,r_s,r_b,train_sigma_gaussians,train_size_blur_kernels);
t3=toc;
save('mdl.mat','mdl','-v7.3');
save('mdls.mat','mdls','-v7.3');

%test models:
tic;
[sigmaOut,kSizeOut]=RUN_SVR_Predict_A(nt,r_t,r_gtt,mdl,mdls,test_sigma_gaussians,test_size_blur_kernels);
t4=toc;
outputs=[test_size_blur_kernels,kSizeOut];

time_of_feature_extraction_from_train_images=t1;
time_of_feature_extraction_from_test_images=t2;
time_of_train_models=t3;
time_of_test_models=t4;
times=[t1;t2;t3;t4];
total_processing_time_minute = sum(times)/60

rks=fix(abs(kSizeOut));
rks1=rks;

for ii=1:nt
    if (mod(rks(ii,1),2)==0)
    rks1(ii,1)=rks(ii,1)+1;
    end
end

kernel_size_out=rks1;

pixel_dif = abs(kernel_size_out-test_size_blur_kernels);
error_percentage = (pixel_dif./test_size_blur_kernels)*100;
status = pixel_dif <= 20;
num_ones = sum(status);
correct_rate = (num_ones / length(pixel_dif))*100

data_table = table( test_size_blur_kernels,kernel_size_out,pixel_dif,error_percentage, 'VariableNames', {'gt_kernel_size','Kernel_size_Out','pixel_dif','error_percentage'});

csv_file_path = 'kernel_size_out.csv';

writetable(data_table, csv_file_path);
