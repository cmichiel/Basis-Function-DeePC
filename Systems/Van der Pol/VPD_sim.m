%% N-step ahead predictor
clc; close all; clear all;



fs = 10;                    % Sampling frequency (samples per second)
Ts = 1/fs;                   % seconds per sample
StopTime = 20;                % seconds
t = (0:Ts:StopTime-Ts)';        % seconds
F = 0.1;                       % Sine wave frequency (hertz)
% A = [0:0.1:0.5, 0.5:-0.1:-0.5,-0.5:0.1:0];
% r = [];
% for j = 1:length(A)
% len = length(t)/length(A);
% r = [r; A(j)*ones(len,1)];
% end  
StopTime = 240;                % seconds
t = (0:Ts:StopTime-Ts)';        % seconds
F = 0.1;
A = [ 0:0.75:1.5,2,1.5:-0.75:-1.5,-2,-2:0.5:2,2,0,0,-2,-2 ]
r = [];
for j = 1:length(A)
len = length(t)/length(A);
r = [r, A(j)*ones(1,len)];
end
%r = 0*ones(1,length(t));

% use this for noiseless case
load('u_data.mat')
load('y_data.mat')

% use this for noisy case 
 % u_data = load('train_1000_noise.mat').input;  %input training data
 % y_data = load('train_1000_noise.mat').output; %output training data
%% load centers and sigmas
% noiseless case
%centers_sigmas = load('train_jointly.mat'); %centers and sigmas trained jointly for all predictors
% noisy case
% centers_sigmas = load('train_jointly_noise.mat'); %centers and sigmas trained jointly for all predictors
load('centers.mat')
load('log_sigmas.mat')
centers_sigmas = struct('centers',centers,'log_sigmas',log_sigmas)
%%
N = 10; %prediction horizon
Tini =3;
n_basis = size(centers,1)
Basis_func = 'Matern52'

k_sim = length(r)-N;
Phi = []; Y = [];

y_ini = ones(Tini,1)*y_data(1)
u_ini = zeros(Tini-1,1)
%% recompute Theta 
for i = 1:length(y_data)-N;
if i == 1
y_ini = [y_ini(2:end);y_data(i)];
u_ini = u_ini;
end
if i >= 2 
y_ini = [y_ini(2:end);y_data(i)];
u_ini = [u_ini(2:end);u_data(i-1)];
end
uf = u_data(i:i+(N-1))';

Phi = [Phi rbf_before_out_layer(centers_sigmas,u_ini,uf,y_ini)];
out_vector = [];
for j = 1:(N)
out_vector = [out_vector; y_data(i+j)];
end

Y = [Y out_vector];
end

Theta = Y*pinv(Phi);

P    = Theta(:,1);
Gamma = Theta(:,2);


%% Build YALMIP SPC problem


centers = double(centers_sigmas.centers);
log_sigmas = double(centers_sigmas.log_sigmas);

Q = 1e2; 
R=  1;

Psi = kron(eye(N), R);
Omega = kron(eye(N), Q);

u = sdpvar(N,1);
y = sdpvar(N,1);
ref = sdpvar(N,1);
u_init = sdpvar(1,1);
%u_ini = sdpvar(Tini-1, 1);
%y_ini = sdpvar(Tini, 1);
gvar = sdpvar(max(size(Phi*Phi')),1);
nl_part = sdpvar(length(centers(:,1)),1);

constraints = [y == Theta*[nl_part;u]];
constraints = [constraints,  -15<=u(1)<=15];
objective = 0;

objective = (y-ref)'*Omega*(y-ref)+(u)'*Psi*(u);
%objective = (y(1:N)-ref(1:N))'*Omega*(y(1:N)-ref(1:N))+(u-u_init)'*Psi*(u-u_init);
%objective = objective +  (y(1)-ref(1))'*Q*(y(1)-ref(1))+(u(1)-u_init )'*R*(u(1)-u_init );

for k = 1:N-1
    %objective = objective +  (y(k+1)-ref(k+1))'*Q*(y(k+1)-ref(k))+(u(k+1)-u(k))'*R*(u(k+1)-u(k));
    constraints = [constraints,  -15<=u(k+1)<=15];
end





Parameters = {nl_part,u_init, ref};
Outputs = {u};

options = sdpsettings('solver', 'quadprog', 'verbose', 0, 'debug', 0);

%options = sdpsettings('solver', 'quadprog', 'verbose', 0, 'debug', 0, 'osqp.eps_abs', 1e-3, 'osqp.eps_rel', 1e-3);
controller = optimizer(constraints, objective, options, Parameters, Outputs);

%% initial conditions
y(1) = 1;
x0 = [1;1]
xk = x0;
xvec1 = [xk];
u_mpc = 0;
uvec = [];
u = [];
th=[];
y_ini = ones(Tini,1)*y(1)
u_ini = zeros(Tini-1,1)

%%
for i = 1:k_sim;
i
tic;
%%% u_ini & y_ini   
if i == 1
y_ini = [y_ini(2:end);y(i)];
u_ini = u_ini;
end
if i >= 2 
y_ini = [y_ini(2:end);y(i)];
u_ini = [u_ini(2:end);uvec(i-1)];
end

clear nl_part;
nl_part = [];
for im = 1:length(centers(:,1))
out = [u_ini;y_ini] - centers(im,:)';  
out = sqrtm(sum(out.^2))./ exp(log_sigmas(im));
out = exp(-1*out^2);
nl_part= [nl_part;out];
end


%%% mpc sol
[Uk ] = controller({nl_part,u_mpc, r(i+1:i+N)'});
u_mpc = Uk(1);
uvec = [uvec Uk(1)];
th=[th;toc];

%State Derivative
mu = 1;
xk = [xk(1) + Ts*xk(2);  xk(2)+Ts*(mu*(1-xk(1)^2)*xk(2)-xk(1)+Uk(1))];
y(i+1) = xk(1);
xvec1 = [xvec1 xk];
end

e = y-r(1:length(y));
%% Plots
fig = figure;
plot(r(1:length(y)),'LineWidth',3);
hold on;
plot(y,'LineWidth',3);
legend('reference','RBF-SPC');
title ('SPC: Reference vs Closed Loop Output');
grid on
saveas(fig,'VDP_RBF-SPC_output_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+'.png')


fig2 = figure;
plot(uvec,'LineWidth',3);
title('SPC: Control Input');
grid on
xlabel('Iterations');
saveas(fig2,'VD_RBF-SPC_input_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+'.png')


figure;
plot(e,'LineWidth',3);
title ('Tracking Error');
grid on;
xlabel('Iterations');

