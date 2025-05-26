function [fe] = l2norm_gaussianFilter(feature)

% feature(isnan(feature))=0; %Removing Infinitiy values
% Normalization of the feature vector using L2-Norm
% feature=feature/sqrt(norm(feature)^2+.001);
% for z=1:length(feature)
%     if feature(z)>0.2
%          feature(z)=0.2;
%     end
% end
% feature=feature/sqrt(norm(feature)^2+.001);

h = fspecial('gaussian',[1 3]);
feature = imfilter(feature,h);
fe=feature/norm(feature);

end

