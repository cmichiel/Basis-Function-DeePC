function [NARXPart] = rbf_NARXPart(RBF,u_ini,uf,y_ini)
%Calculate NARX component of Phi
rbf = [];

for i = 1:length(RBF.centersNARX(:,1))
out = ([u_ini;y_ini;uf]-RBF.data_mean')./RBF.data_std' - RBF.centersNARX(i,:)';  
out = sqrtm(sum(out.^2))./ exp(RBF.log_sigmasNARX(i));
if string(RBF.Basis_funcNARX) == 'gaussian'
    out = exp(-1*out^2);
elseif string(Basis_funcNARX) == 'spline'
    out = (out.^2 * log(out + 1));
elseif string(Basis_funcNARX) == 'inverse multiquadratic'
    out = 1 /( 1 + out^2);
elseif string(Basis_funcNARX) == 'matern52'
   out = (1 + sqrt(5) * out + (5/3) * out .^2) .* exp(-sqrt(5) * out );
  
end

rbf = [rbf;out];
end
NARXPart = rbf;

end