function [sigmaOut,kSizeOut]=RUN_SVR_Predict_A(nt,r_t,r_gtt,mdl,mdls,test_sigma_gaussians,test_size_blur_kernels)             
  
    num_bins=360;

    %r_t from blur images for test:
    r1tt=zeros(nt,num_bins);
    r2tt=zeros(nt,num_bins);
    r3tt=zeros(nt,num_bins);
    r4tt=zeros(nt,num_bins);
    r5tt=zeros(nt,num_bins);

    for it=1:nt
        r1tt(it,:)=r_t{it,1};
        r2tt(it,:)=r_t{it,2};
        r3tt(it,:)=r_t{it,3};
        r4tt(it,:)=r_t{it,4};
        r5tt(it,:)=r_t{it,5};
    end

     %scaling:
     Rt(:,:,1)=r1tt;
     Rt(:,:,2)=r2tt;
     Rt(:,:,3)=r3tt;
     Rt(:,:,4)=r4tt;
     Rt(:,:,5)=r5tt;
     Rts=rescale(Rt,-1,1);
     r1t=Rts(:,:,1);
     r2t=Rts(:,:,2);
     r3t=Rts(:,:,3);
     r4t=Rts(:,:,4);
     r5t=Rts(:,:,5);
    
    %r_gtt from sharp images for test:
    r1g=zeros(nt,num_bins);
    r2g=zeros(nt,num_bins);
    r3g=zeros(nt,num_bins);
    r4g=zeros(nt,num_bins);
    r5g=zeros(nt,num_bins);

    for igtt=1:nt
        r1g(igtt,:)=r_gtt{igtt,1};
        r2g(igtt,:)=r_gtt{igtt,2};
        r3g(igtt,:)=r_gtt{igtt,3};
        r4g(igtt,:)=r_gtt{igtt,4};
        r5g(igtt,:)=r_gtt{igtt,5};
    end

     %scaling:
     Rg(:,:,1)=r1g;
     Rg(:,:,2)=r2g;
     Rg(:,:,3)=r3g;
     Rg(:,:,4)=r4g;
     Rg(:,:,5)=r5g;
     Rgs=rescale(Rg,-1,1);
     r1gtt=Rgs(:,:,1);
     r2gtt=Rgs(:,:,2);
     r3gtt=Rgs(:,:,3);
     r4gtt=Rgs(:,:,4);
     r5gtt=Rgs(:,:,5);
      
    %t = test : blur, gtt = groundtruth of test: sharp
    
    for j=1:num_bins
        
        [out_r1(:,j) accuracy1(:,j) prob1(:,j)]=svmpredict_(r1gtt(:,j),[r4t(:,j),r5t(:,j)],mdl{j,1});
              
        r11=out_r1(:,j);
            
        [out_r2(:,j) accuracy2(:,j) prob2(:,j)]=svmpredict_(r2gtt(:,j),[r11,r4t(:,j),r5t(:,j)],mdl{j,2});           
        
        [out_r3(:,j) accuracy3(:,j) prob3(:,j)]=svmpredict_(r3gtt(:,j),[r11,r4t(:,j),r5t(:,j)],mdl{j,3});
        
        [out_r1r(:,j) accuracy0(:,j) prob0(:,j)]=svmpredict_(r1gtt(:,j),[out_r2(:,j),out_r3(:,j),r4t(:,j),r5t(:,j)],mdl{j,4});
        
        r12=out_r1r(:,j);
        
        r1final(:,j) = r12;
        r2final(:,j) = out_r2(:,j);
        r3final(:,j) = out_r3(:,j);
    end 

    delta_r1t=r1final-r1t;
    delta_r2t=r2final-r2t;
    delta_r3t=r3final-r3t;

    for j=1:num_bins    
                                                           
        [sigmaOut(:,j) accuracys1(:,j) probs1(:,j)]=svmpredict_(test_sigma_gaussians,[delta_r1t(:,j),delta_r2t(:,j),delta_r3t(:,j)],mdl{j,5});
        
    end     
    
    [kSizeOut a p]=svmpredict_(test_size_blur_kernels,sigmaOut,mdls);
    
end