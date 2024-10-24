function [KoopmanPart] = rbf_KoopmanPart(RBF,u_ini,y_ini)
%UNTITLED8 Summary of this function goes here


%Calculate  Koopman component of Phi
rbf = [];

for i = 1:length(RBF.centersKoopman(:,1))
out = ([u_ini;y_ini]-RBF.data_mean(1:2*RBF.Tini-1)')./RBF.data_std(1:2*RBF.Tini-1)' - RBF.centersKoopman(i,:)';  
out = sqrtm(sum(out.^2))./ exp(RBF.log_sigmasKoopman(i));
if string(RBF.Basis_funcKoopman) == 'gaussian'
    out = exp(-1*out^2);
elseif string(RBF.Basis_funcKoopman) == 'spline'
    out = (out.^2 * log(out + 1));
elseif string(RBF.Basis_funcKoopman) == 'inverse multiquadratic'
    out = 1 /( 1 + out^2);
elseif string(RBF.Basis_funcKoopman) == 'matern52'
   out = (1 + sqrt(5) * out + (5/3) * out .^2) .* exp(-sqrt(5) * out );
  
end

rbf = [rbf;out];
end
KoopmanPart = rbf;

end
