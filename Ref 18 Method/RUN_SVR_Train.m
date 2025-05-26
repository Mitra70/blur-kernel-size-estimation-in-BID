function [mdl]=RUN_SVR_Train(nb,r_s,r_b,sig_g,rep_sk_tr)
    
    tav=0.001; %threshold
    maxItr=10; %max iteration
    num_bins=180;

    mdl=cell(num_bins,5); %4 models + sigma model

    %r from blur images for train:
    r1bb=zeros(nb,num_bins);
    r2bb=zeros(nb,num_bins);
    r3bb=zeros(nb,num_bins);
    r4bb=zeros(nb,num_bins);
    r5bb=zeros(nb,num_bins);

    for ib=1:nb
        r1bb(ib,:)=r_b{ib,1};
        r2bb(ib,:)=r_b{ib,2};
        r3bb(ib,:)=r_b{ib,3};
        r4bb(ib,:)=r_b{ib,4};
        r5bb(ib,:)=r_b{ib,5};
    end

     %scaling:
     Rb(:,:,1)=r1bb;
     Rb(:,:,2)=r2bb;
     Rb(:,:,3)=r3bb;
     Rb(:,:,4)=r4bb;
     Rb(:,:,5)=r5bb;
     Rbs=rescale(Rb,-1,1);
     r1b=Rbs(:,:,1);
     r2b=Rbs(:,:,2);
     r3b=Rbs(:,:,3);
     r4b=Rbs(:,:,4);
     r5b=Rbs(:,:,5);
    
    %r from sharp images for train:
    r1ss=zeros(nb,num_bins);
    r2ss=zeros(nb,num_bins);
    r3ss=zeros(nb,num_bins);
    r4ss=zeros(nb,num_bins);
    r5ss=zeros(nb,num_bins);

    for is=1:nb
        r1ss(is,:)=r_s{is,1};
        r2ss(is,:)=r_s{is,2};
        r3ss(is,:)=r_s{is,3};
        r4ss(is,:)=r_s{is,4};
        r5ss(is,:)=r_s{is,5};
    end
   
     %scaling:
     Rs(:,:,1)=r1ss;
     Rs(:,:,2)=r2ss;
     Rs(:,:,3)=r3ss;
     Rs(:,:,4)=r4ss;
     Rs(:,:,5)=r5ss;
     Rss=rescale(Rs,-1,1);
     r1s=Rss(:,:,1);
     r2s=Rss(:,:,2);
     r3s=Rss(:,:,3);
     r4s=Rss(:,:,4);
     r5s=Rss(:,:,5);

    %s = sharp, b = blur

    for j=1:num_bins

        itr=0;
        
        mdl{j,1}=svmtrain_(r1s(:,j),[r4b(:,j),r5b(:,j)],'-s 3 -t 0');
        [out_r1(:,j) a1(:,j) p1(:,j)]=svmpredict_(r1s(:,j),[r4b(:,j),r5b(:,j)],mdl{j,1});
      
        r11=r1s(:,j);
        r12=out_r1(:,j);

        while ( (norm(r12-r11))^2>tav && itr<maxItr )
            r11=r12;

            mdl{j,2}=svmtrain_(r2s(:,j),[r11,r4b(:,j),r5b(:,j)],'-s 3 -t 0');
            [out_r2(:,j) a2(:,j) p2(:,j)]=svmpredict_(r2s(:,j),[r11,r4b(:,j),r5b(:,j)],mdl{j,2});
            
            mdl{j,3}=svmtrain_(r3s(:,j),[r11,r4b(:,j),r5b(:,j)],'-s 3 -t 0');
            [out_r3(:,j) a3(:,j) p3(:,j)]=svmpredict_(r3s(:,j),[r11,r4b(:,j),r5b(:,j)],mdl{j,3});
            
            mdl{j,4}=svmtrain_(r1s(:,j),[out_r2(:,j),out_r3(:,j),r4b(:,j),r5b(:,j)],'-s 3 -t 0');
            [out_r1r(:,j) a0(:,j) p0(:,j)]=svmpredict_(r1s(:,j),[out_r2(:,j),out_r3(:,j),r4b(:,j),r5b(:,j)],mdl{j,4});
            
            r12=out_r1r(:,j);

            itr=itr+1;
        end %for while

        r1final(:,j) = r12;
        r2final(:,j) = out_r2(:,j);
        r3final(:,j) = out_r3(:,j);
        
    end
    
    delta_r1b=r1s-r1b;
    delta_r2b=r2s-r2b;
    delta_r3b=r3s-r3b;

    for j=1:num_bins 
                           %rep_sk_tr
        mdl{j,5}=svmtrain_(sig_g,[delta_r1b(:,j),delta_r2b(:,j),delta_r3b(:,j)],'-s 3 -t 0');
           
    end
end