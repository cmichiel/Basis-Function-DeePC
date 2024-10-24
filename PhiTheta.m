function [RBF] = PhiTheta(RBF)
%Construct Theta and Phi for Data Driven Control

u_data = RBF.u_data;
y_data = RBF.y_data;
KoopmanPart = [];
uf1 = [];
NARXPart = [];
ResPart = [];
Phi = []; 
Y = [];

y_ini = ones(RBF.Tini,1)*y_data(1)
u_ini = zeros(RBF.Tini-1,1)

%   Loop to Construct Phi
for i = 1:length(y_data)-RBF.N
    if i == 1
        y_ini = [y_ini(2:end);y_data(i)];
        u_ini = u_ini;
    end
    if i >= 2 
        y_ini = [y_ini(2:end);y_data(i)];
        u_ini = [u_ini(2:end);u_data(i-1)];
    end
    uf = u_data(i:i+(RBF.N-1))'; 
    
    
    if  strcmp(RBF.BasisStructure, 'Koopman') || strcmp(RBF.BasisStructure, 'KoopmanResnet') || strcmp(RBF.BasisStructure, 'KoopmanNARX')
        KoopmanPart = rbf_KoopmanPart(RBF,u_ini,y_ini);
        uf1 = (uf-RBF.data_mean(2*RBF.Tini:2*RBF.Tini-1+RBF.N)')./RBF.data_std(2*RBF.Tini:2*RBF.Tini-1+RBF.N)';
    end
    
    if  strcmp(RBF.BasisStructure, 'NARX') || strcmp(RBF.BasisStructure, 'NARXResnet') || strcmp(RBF.BasisStructure, 'KoopmanNARX')
        NARXPart = rbf_NARXPart(RBF,u_ini,uf,y_ini);
    end
    
    if  strcmp(RBF.BasisStructure, 'KoopmanResnet') || strcmp(RBF.BasisStructure, 'NARXResnet')
        ResPart = ([u_ini;y_ini;uf]-RBF.data_mean')./RBF.data_std';
    end
    
    if  strcmp(RBF.BasisStructure, 'KoopmanResnet') || strcmp(RBF.BasisStructure, 'NARXResnet')
        Phi_part = [KoopmanPart; NARXPart; ResPart];
    else
        Phi_part = [KoopmanPart; NARXPart; uf1];
    end

    Phi = [Phi Phi_part];
    
    out_vector = [];
    for j = 1:(RBF.N)
        out_vector = [out_vector; y_data(i+j)];
    end
    Y = [Y out_vector];
end
RBF.Y = Y;
RBF.Phi = Phi;
RBF.Theta = Y*pinv(Phi);
end