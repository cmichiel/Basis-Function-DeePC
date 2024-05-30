function [phi1_nn] =rbf_before_out_layer(param,uu,u,y)
centers = double(param.centers);
log_sigmas = double(param.log_sigmas);
rbf = [];
for i = 1:length(centers)
out = [uu;y] - centers(i,:)';  
out = norm(out,2)./ exp(log_sigmas(i));
out = exp(-1*out^2);
rbf = [rbf;out];
end
phi1_nn = [rbf;u];
end