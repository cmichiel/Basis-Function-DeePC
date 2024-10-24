function [Controller] = SelectController(RBF, Controller)
%Construct the desires Control structure
%   Detailed explanation goes here
Controller.Psi = kron(eye(RBF.N), Controller.R);
Controller.Omega = kron(eye(RBF.N), Controller.Q);

%Construct YALMIP Variables
u = sdpvar(RBF.N,1);
y = sdpvar(RBF.N,1);
ref = sdpvar(RBF.N,1);
u_init = sdpvar(1,1);
u_ini_var = sdpvar(RBF.Tini-1, 1);
y_ini_var = sdpvar(RBF.Tini, 1);

objective = 0;

if Controller.Type == 'SPC'
    if strcmp(RBF.BasisStructure, 'Koopman')
        nl_part = sdpvar(length(RBF.centersKoopman(:,1)),1);
        constraints = [y == RBF.Theta*[nl_part;(u-RBF.data_mean(2*RBF.Tini:2*RBF.Tini-1+RBF.N)')./RBF.data_std(2*RBF.Tini:2*RBF.Tini-1+RBF.N)']];
        Parameters = {nl_part, u_init, ref,u_ini_var,y_ini_var};
   
    elseif strcmp(RBF.BasisStructure, 'KoopmanResnet')
        nl_part = sdpvar(length(RBF.centersKoopman(:,1)),1);
        constraints = [y == RBF.Theta*[nl_part;  ([u_ini_var; y_ini_var;u]-RBF.data_mean')./RBF.data_std'; ]];
        Parameters = {nl_part, u_init, ref,u_ini_var,y_ini_var};
    
    elseif strcmp(RBF.BasisStructure, 'NARX')
         constraints = [y == RBF.Theta*[rbf_NARXPart(RBF,u_ini_var,u,y_ini_var)]];
         Parameters = {u_init, ref,u_ini_var,y_ini_var};
    
    elseif strcmp(RBF.BasisStructure, 'NARXResnet')
        constraints = [y == RBF.Theta*[rbf_NARXPart(RBF,u_ini_var,u,y_ini_var); ([u_ini_var; y_ini_var;u]-RBF.data_mean')./RBF.data_std';  ]];
        Parameters = {u_init, ref,u_ini_var,y_ini_var};
    
    elseif strcmp(RBF.BasisStructure, 'KoopmanNARX')
        nl_part = sdpvar(length(RBF.centersKoopman(:,1)),1);
        constraints = [y == RBF.Theta*[nl_part;(u-RBF.data_mean(2*RBF.Tini:2*RBF.Tini-1+RBF.N)')./RBF.data_std(2*RBF.Tini:2*RBF.Tini-1+RBF.N)';
        rbf_NARXPart(RBF,u_ini_var,u,y_ini_var)]];
        Parameters = {nl_part, u_init, ref,u_ini_var,y_ini_var};
    end
    
elseif Controller.Type == 'DeePC-R2'
    objective = objective + lambda*norm(gvar,2)^2;
    gvar = sdpvar(length(RBF.u_data)-RBF.N,1);
    if strcmp(RBF.BasisStructure, 'Koopman')
        nl_part = sdpvar(length(RBF.centersKoopman(:,1)),1);
        constraints = [RBF.Phi*gvar == 0];
        constraints = [constraints,  RBF.Theta*[nl_part; (u-RBF.data_mean(2*RBF.Tini:2*RBF.Tini-1+RBF.N)')./RBF.data_std(2*RBF.Tini:2*RBF.Tini-1+RBF.N)']+RBF.Y*gvar==y];
    
    elseif strcmp(RBF.BasisStructure, 'KoopmanResnet')
        nl_part = sdpvar(length(RBF.centersKoopman(:,1)),1);
        constraints = [RBF.Phi*gvar == 0];
        constraints = [constraints,  RBF.Theta*[nl_part;  (([u_ini_var; y_ini_var;u]-RBF.data_mean')./RBF.data_std'); ]+RBF.Y*gvar==y];
    
    elseif strcmp(RBF.BasisStructure, 'NARX')
        constraints = [RBF.Phi*gvar == 0];
        constraints = [constraints,  RBF.Theta*[rbf_NARXPart(RBF,u_ini_var,u,y_ini_var)]+RBF.Y*gvar==y];
    
    elseif strcmp(RBF.BasisStructure, 'NARXResnet')
        constraints = [RBF.Phi*gvar == 0];
        constraints = [constraints,  RBF.Theta*[rbf_NARXPart(RBF,u_ini_var,u,y_ini_var); (([u_ini_var; y_ini_var;u]-RBF.data_mean')./RBF.data_std')]+RBF.Y*gvar==y];
    
    elseif strcmp(RBF.BasisStructure, 'KoopmanNARX')
        nl_part = sdpvar(length(RBF.centersKoopman(:,1)),1);
        constraints = [RBF.Phi*gvar == 0];
        constraints = [constraints, RBF.Theta*[nl_part; (u-RBF.data_mean(2*RBF.Tini:2*RBF.Tini-1+RBF.N)')./RBF.data_std(2*RBF.Tini:2*RBF.Tini-1+RBF.N);
        rbf_NARXPart(RBF,u_ini_var,u,y_ini_var)]+RBF.Y*gvar==y];
    end
end


objective = (y(1:RBF.N)-ref(1:RBF.N))'*Controller.Omega*(y(1:RBF.N)-ref(1:RBF.N))+(u(1)-u_init)'*Controller.Rd*(u(1)-u_init) + u(RBF.N)'*Controller.R*u(RBF.N);

for k = 1:RBF.N-1
    objective = objective + u(k)'*Controller.R*u(k);
    objective = objective +(u(k+1)-u(k))'*Controller.Rd*(u(k+1)-u(k));
    constraints = [constraints,  -15<=u(k+1)<=15];
end

objective = objective+(y(RBF.N)-ref(RBF.N))'*Controller.S*(y(RBF.N)-ref(RBF.N));
Outputs = {u,y};

Controller.controller = optimizer(constraints, objective,[], Parameters, Outputs);

end