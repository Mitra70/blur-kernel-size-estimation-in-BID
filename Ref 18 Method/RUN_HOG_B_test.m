function [r_t,r_gtt]=RUN_HOG_B_test(nt,ngtt,nkt,kernel_type)

level=6; %number of pyramid levels
num_bins=180;

hog_t=cell(nt,level);
fea_t=cell(nt,level);
r_t=cell(nt,level-1);

hog_gtt=cell(ngtt,level);
fea_gtt=cell(ngtt,level);
r_gtt1=cell(ngtt,level-1);
r_gtt=cell(ngtt*nkt,level-1);

kt=1; %count number of blur images.
kgtt=1; %count number of sharp images.

switch kernel_type
    case 'gaussian_symmetric'
        test_blur_path_pattern = '../dataset/gaussian blur test images with sym gk 2/tb (%d).png';
    case 'gaussian_without_tilt'
        test_blur_path_pattern = '../dataset/gaussian blur test images without tilt 2/tb (%d).png';
    case 'gaussian_tilt'
        test_blur_path_pattern = '../dataset/gaussian blur test images with tilt 2/tb (%d).png';
    case 'non_gaussian'
        test_blur_path_pattern = '../dataset/blur test images with non gk 2/tb (%d).png';
    otherwise
        error('Unknown kernel type.');
end

for it=1:nt
    
%Blurry Test Input:
filename_b = sprintf(test_blur_path_pattern, it);

img_b=imread(filename_b);
if numel(size(img_b))>2
img_b_g=im2double(rgb2gray(img_b));
else
img_b_g=im2double(img_b);
end

%Image pyramid reduction and expansion:
It1 = img_b_g;
It2 = impyramid(It1, 'reduce');
It3 = impyramid(It2, 'reduce');
It4 = impyramid(It3, 'reduce');
It5 = impyramid(It4, 'reduce');
It6 = impyramid(It5, 'reduce');

%HOG:
[ht1,vt1] = extractHOGFeatures(It1,'Cellsize',[size(It1,1) size(It1,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[ht2,vt2] = extractHOGFeatures(It2,'Cellsize',[size(It2,1) size(It2,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[ht3,vt3] = extractHOGFeatures(It3,'Cellsize',[size(It3,1) size(It3,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[ht4,vt4] = extractHOGFeatures(It4,'Cellsize',[size(It4,1) size(It4,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[ht5,vt5] = extractHOGFeatures(It5,'Cellsize',[size(It5,1) size(It5,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[ht6,vt6] = extractHOGFeatures(It6,'Cellsize',[size(It6,1) size(It6,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);

fea_t{kt,1}=ht1;
fea_t{kt,2}=ht2;
fea_t{kt,3}=ht3;
fea_t{kt,4}=ht4;
fea_t{kt,5}=ht5;
fea_t{kt,6}=ht6;

for L=1:6
    idt=find(fea_t{kt,L}==0);
    if isempty(idt)==0
        fea_t{kt,L}(idt)=10^(-10);
    end
end

hog_t{kt,1} = l2norm_gaussianFilter(fea_t{kt,1});
hog_t{kt,2} = l2norm_gaussianFilter(fea_t{kt,2});
hog_t{kt,3} = l2norm_gaussianFilter(fea_t{kt,3});
hog_t{kt,4} = l2norm_gaussianFilter(fea_t{kt,4});
hog_t{kt,5} = l2norm_gaussianFilter(fea_t{kt,5});
hog_t{kt,6} = l2norm_gaussianFilter(fea_t{kt,6});

r_t{kt,1} = double(log(hog_t{kt,6})-log(hog_t{kt,1}));
r_t{kt,2} = double(log(hog_t{kt,6})-log(hog_t{kt,2}));
r_t{kt,3} = double(log(hog_t{kt,6})-log(hog_t{kt,3}));
r_t{kt,4} = double(log(hog_t{kt,6})-log(hog_t{kt,4}));
r_t{kt,5} = double(log(hog_t{kt,6})-log(hog_t{kt,5}));

kt=kt+1;

end %for it

q=1;

for igtt=1:ngtt
    
%groundtruth of Blurry Test Input:
filename_gtb=sprintf('../dataset/test sharp images 2/ts (%d).png',igtt);
img_gt_b=imread(filename_gtb);
if numel(size(img_gt_b))>2
img_gt_b_g=im2double(rgb2gray(img_gt_b));
else
img_gt_b_g=im2double(img_gt_b);
end

%Image pyramid reduction and expansion:
Igtt1 = img_gt_b_g;
Igtt2 = impyramid(Igtt1, 'reduce');
Igtt3 = impyramid(Igtt2, 'reduce');
Igtt4 = impyramid(Igtt3, 'reduce');
Igtt5 = impyramid(Igtt4, 'reduce');
Igtt6 = impyramid(Igtt5, 'reduce');

%HOG:
[hgtt1,vgtt1] = extractHOGFeatures(Igtt1,'Cellsize',[size(Igtt1,1) size(Igtt1,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[hgtt2,vgtt2] = extractHOGFeatures(Igtt2,'Cellsize',[size(Igtt2,1) size(Igtt2,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[hgtt3,vgtt3] = extractHOGFeatures(Igtt3,'Cellsize',[size(Igtt3,1) size(Igtt3,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[hgtt4,vgtt4] = extractHOGFeatures(Igtt4,'Cellsize',[size(Igtt4,1) size(Igtt4,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[hgtt5,vgtt5] = extractHOGFeatures(Igtt5,'Cellsize',[size(Igtt5,1) size(Igtt5,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[hgtt6,vgtt6] = extractHOGFeatures(Igtt6,'Cellsize',[size(Igtt6,1) size(Igtt6,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);

fea_gtt{kgtt,1}=hgtt1;
fea_gtt{kgtt,2}=hgtt2;
fea_gtt{kgtt,3}=hgtt3;
fea_gtt{kgtt,4}=hgtt4;
fea_gtt{kgtt,5}=hgtt5;
fea_gtt{kgtt,6}=hgtt6;

for L=1:6
    idgtt=find(fea_gtt{kgtt,L}==0);
    if isempty(idgtt)==0
        fea_gtt{kgtt,L}(idgtt)=10^(-10);
    end
end

hog_gtt{kgtt,1} = l2norm_gaussianFilter(fea_gtt{kgtt,1});
hog_gtt{kgtt,2} = l2norm_gaussianFilter(fea_gtt{kgtt,2});
hog_gtt{kgtt,3} = l2norm_gaussianFilter(fea_gtt{kgtt,3});
hog_gtt{kgtt,4} = l2norm_gaussianFilter(fea_gtt{kgtt,4});
hog_gtt{kgtt,5} = l2norm_gaussianFilter(fea_gtt{kgtt,5});
hog_gtt{kgtt,6} = l2norm_gaussianFilter(fea_gtt{kgtt,6});

r_gtt1{kgtt,1} = double(log(hog_gtt{kgtt,6})-log(hog_gtt{kgtt,1}));
r_gtt1{kgtt,2} = double(log(hog_gtt{kgtt,6})-log(hog_gtt{kgtt,2}));
r_gtt1{kgtt,3} = double(log(hog_gtt{kgtt,6})-log(hog_gtt{kgtt,3}));
r_gtt1{kgtt,4} = double(log(hog_gtt{kgtt,6})-log(hog_gtt{kgtt,4}));
r_gtt1{kgtt,5} = double(log(hog_gtt{kgtt,6})-log(hog_gtt{kgtt,5}));

for g=q:(q+nkt-1)
r_gtt(g,:)=r_gtt1(kgtt,:);
end

q=q+nkt;

kgtt=kgtt+1;

end %for igtt

end