function [r_s,r_b]=RUN_HOG_B_S_A(nb,nk,kernel_type)

level=6; %number of pyramid levels
num_bins=360;

hog_b=cell(nb,level);
fea_b=cell(nb,level);
r_b=cell(nb,level-1);

hog_s=cell(nb,level);
fea_s=cell(nb,level);
r_s=cell(nb,level-1);

cb=1; %count number of blur images.
cs=1; %count number of sharp images.
q=1;

switch kernel_type
    case 'gaussian_symmetric'
        blur_path_pattern = '../dataset/gaussian blur images with sym gk_/gb_%d.png';
    case 'gaussian_without_tilt'
        blur_path_pattern = '../dataset/gaussian blur images without tilt_/gb_%d.png';
    case 'gaussian_tilt'
        blur_path_pattern = '../dataset/gaussian blur images with tilt_/gb_%d.png';
    case 'non_gaussian'
        blur_path_pattern = '../dataset/blur images with non gk_/gb_%d.png';
    otherwise
        error('Unknown kernel type.');
end

for ib=1:nb

%blur Input:
filename_b = sprintf(blur_path_pattern, ib);

img_b=imread(filename_b);
if numel(size(img_b))>2
img_b_g=im2double(rgb2gray(img_b));
else
img_b_g=im2double(img_b);
end

%Image pyramid reduction and expansion:
Ib1 = img_b_g;
Ib2 = impyramid(Ib1, 'reduce');
Ib3 = impyramid(Ib2, 'reduce');
Ib4 = impyramid(Ib3, 'reduce');
Ib5 = impyramid(Ib4, 'reduce');
Ib6 = impyramid(Ib5, 'reduce');

%HOG:
[hb1,vb1] = extractHOGFeatures(Ib1,'Cellsize',[size(Ib1,1) size(Ib1,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[hb2,vb2] = extractHOGFeatures(Ib2,'Cellsize',[size(Ib2,1) size(Ib2,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[hb3,vb3] = extractHOGFeatures(Ib3,'Cellsize',[size(Ib3,1) size(Ib3,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[hb4,vb4] = extractHOGFeatures(Ib4,'Cellsize',[size(Ib4,1) size(Ib4,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[hb5,vb5] = extractHOGFeatures(Ib5,'Cellsize',[size(Ib5,1) size(Ib5,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[hb6,vb6] = extractHOGFeatures(Ib6,'Cellsize',[size(Ib6,1) size(Ib6,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);

fea_b{cb,1}=hb1;
fea_b{cb,2}=hb2;
fea_b{cb,3}=hb3;
fea_b{cb,4}=hb4;
fea_b{cb,5}=hb5;
fea_b{cb,6}=hb6;

for L=1:level
    idb=find(fea_b{cb,L}==0);
    if isempty(idb)==0
        fea_b{cb,L}(idb)=10^(-10);
    end
end

hog_b{cb,1} = l2norm_gaussianFilter(fea_b{cb,1});
hog_b{cb,2} = l2norm_gaussianFilter(fea_b{cb,2});
hog_b{cb,3} = l2norm_gaussianFilter(fea_b{cb,3});
hog_b{cb,4} = l2norm_gaussianFilter(fea_b{cb,4});
hog_b{cb,5} = l2norm_gaussianFilter(fea_b{cb,5});
hog_b{cb,6} = l2norm_gaussianFilter(fea_b{cb,6});

r_b{cb,1} = double(log(hog_b{cb,6})-log(hog_b{cb,1}));
r_b{cb,2} = double(log(hog_b{cb,6})-log(hog_b{cb,2}));
r_b{cb,3} = double(log(hog_b{cb,6})-log(hog_b{cb,3}));
r_b{cb,4} = double(log(hog_b{cb,6})-log(hog_b{cb,4}));
r_b{cb,5} = double(log(hog_b{cb,6})-log(hog_b{cb,5}));

cb=cb+1;

end %for ib

for is=1:(nb/nk)
    
%gt Input:
filename_gt=sprintf('../dataset/gt images_/%d_gt.png',is);
img_gt=imread(filename_gt);
if numel(size(img_gt))>2
img_gt_g=im2double(rgb2gray(img_gt));
else
img_gt_g=im2double(img_gt);
end

%Image pyramid reduction and expansion:
Is1 = img_gt_g;
Is2 = impyramid(Is1, 'reduce');
Is3 = impyramid(Is2, 'reduce');
Is4 = impyramid(Is3, 'reduce');
Is5 = impyramid(Is4, 'reduce');
Is6 = impyramid(Is5, 'reduce');

%HOG:
[hs1,vs1] = extractHOGFeatures(Is1,'Cellsize',[size(Is1,1) size(Is1,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[hs2,vs2] = extractHOGFeatures(Is2,'Cellsize',[size(Is2,1) size(Is2,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[hs3,vs3] = extractHOGFeatures(Is3,'Cellsize',[size(Is3,1) size(Is3,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[hs4,vs4] = extractHOGFeatures(Is4,'Cellsize',[size(Is4,1) size(Is4,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[hs5,vs5] = extractHOGFeatures(Is5,'Cellsize',[size(Is5,1) size(Is5,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);
[hs6,vs6] = extractHOGFeatures(Is6,'Cellsize',[size(Is6,1) size(Is6,2)],'BlockSize',[1 1],'NumBins',num_bins,'UseSignedOrientation',false);

fea_s{cs,1}=hs1;
fea_s{cs,2}=hs2;
fea_s{cs,3}=hs3;
fea_s{cs,4}=hs4;
fea_s{cs,5}=hs5;
fea_s{cs,6}=hs6;

for L=1:level
    ids=find(fea_s{cs,L}==0);
    if isempty(ids)==0
        fea_s{cs,L}(ids)=10^(-10);
    end
end

hog_s{cs,1} = l2norm_gaussianFilter(fea_s{cs,1});
hog_s{cs,2} = l2norm_gaussianFilter(fea_s{cs,2});
hog_s{cs,3} = l2norm_gaussianFilter(fea_s{cs,3});
hog_s{cs,4} = l2norm_gaussianFilter(fea_s{cs,4});
hog_s{cs,5} = l2norm_gaussianFilter(fea_s{cs,5});
hog_s{cs,6} = l2norm_gaussianFilter(fea_s{cs,6});

r_s1{cs,1} = double(log(hog_s{cs,6})-log(hog_s{cs,1}));
r_s1{cs,2} = double(log(hog_s{cs,6})-log(hog_s{cs,2}));
r_s1{cs,3} = double(log(hog_s{cs,6})-log(hog_s{cs,3}));
r_s1{cs,4} = double(log(hog_s{cs,6})-log(hog_s{cs,4}));
r_s1{cs,5} = double(log(hog_s{cs,6})-log(hog_s{cs,5}));

for g=q:(q+nk-1)
r_s(g,:)=r_s1(cs,:);
end
q=q+nk;

cs=cs+1;

end %for is

end