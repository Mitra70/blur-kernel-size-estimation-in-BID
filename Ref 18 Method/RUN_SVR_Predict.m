function [TRsig,sigma,xj,yj,fi]=RUN_SVR_Predict(nt,r_t,r_gtt,mdl,sig_t,rep_sk_t)             
  
    num_bins=180;

    %r from blur images for test:
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

    %r from sharp images for test:
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

    %t = test, gtt = groundtruth of test
    
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
                                                           %rep_sk_t
        [sig(:,j) accuracys1(:,j) probs1(:,j)]=svmpredict_(sig_t,[delta_r1t(:,j),delta_r2t(:,j),delta_r3t(:,j)],mdl{j,5});
        
    end 

    TRsig=sig';

    %estimate a 2D elliptical kernel shape in Euclidean space:
    for i=1:nt
        xj(1:num_bins,i)=TRsig(1:num_bins,i).*(cos((1:180)*pi/180))';
        yj(1:num_bins,i)=TRsig(1:num_bins,i).*(sin((1:180)*pi/180))';

        points=[xj(1:num_bins,i),yj(1:num_bins,i)];
        P1=points;
        P2=-P1;
        P=[P1;P2];
        filterData = ellipseDataFilter_RANSAC(P);
        % para = funcEllipseFit_direct(filterData);
        fisher = funcEllipseFit_BFisher(filterData(:,1),filterData(:,2));

        xx=P(:,1); yy=P(:,2);
        xx2=filterData(:,1); yy2=filterData(:,2);
        oo=figure('visible','off');
        hold on;
        plot(xx,yy,'r.',xx2,yy2,'bo');
        title(['sigma:',num2str(sig_t(i)),' & kernel size:',num2str(rep_sk_t(i))]);
        legend('Outlier points','Inlier points');
        plotellipse([fisher(1);fisher(2)], fisher(3), fisher(4), fisher(5));
        hold off;

%         print(oo,'-dpng',strcat('ellipses with sym gk/ellipse_',num2str(i),'.png'));
%         savefig(oo,strcat('ellipses with sym gk/ellipse_',num2str(i),'.fig'));

%         print(oo,'-dpng',strcat('ellipses without tilt/ellipse_',num2str(i),'.png'));
%         savefig(oo,strcat('ellipses without tilt/ellipse_',num2str(i),'.fig'));

%         print(oo,'-dpng',strcat('ellipses with tilt/ellipse_',num2str(i),'.png'));
%         savefig(oo,strcat('ellipses with tilt/ellipse_',num2str(i),'.fig'));

%         print(oo,'-dpng',strcat('ellipses_non gk/ellipse_',num2str(i),'.png'));
%         savefig(oo,strcat('ellipses_non gk/ellipse_',num2str(i),'.fig'));

        x0=fisher(1); y0=fisher(2); aa=fisher(3); bb=fisher(4); teta=fisher(5);

        A=(aa^2)*((sin(teta))^2)+(bb^2)*((cos(teta))^2);
        B=2*(bb^2-aa^2)*sin(teta)*cos(teta);
        C=(aa^2)*((cos(teta))^2)+(bb^2)*((sin(teta))^2);
        D=-2*A*x0-B*y0; 
        E=-1*B*x0-2*C*y0;
        F=A*(x0^2)+B*x0*y0+C*(y0^2)-1*(aa^2)*(bb^2);
        F=-F;
        prms=[A B C D E F]; 
        n_prms=prms/F;

        %paper equ: ax^2+2cxy+by^2=1:
        %A=[xj(1:num_bins,i).^2,2*xj(1:num_bins,i).*yj(1:num_bins,i),yj(1:num_bins,i).^2];
        %A*fi=ONE:
        %fi(1:3,i)=pinv(A)*ONE;

        fi(i,1)=n_prms(1);
        fi(i,2)=n_prms(2)/2; 
        fi(i,3)=n_prms(3);

        a=fi(i,1);
        c=fi(i,2);
        b=fi(i,3);

        sigx(i,1)=2*sqrt(2/(abs(a+b+sqrt((a-b)^2+4*c^2))));
        sigy(i,1)=2*sqrt(2/(abs(a+b-sqrt((a-b)^2+4*c^2))));

        close all
    end 

    sigma=[(sigx),(sigy)];
end