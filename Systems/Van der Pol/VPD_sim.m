%% N-step ahead predictor
clc; close all; clear all;

M = 1.   ;       % mass of the pendulum
L = 1.   ;       % lenght of the pendulum
b = 0.1  ;       % friction coefficient
g = 9.81  ;      % acceleration of gravity
J = 1/3*M*L^2 ;  % moment of inertia
Ts = 1/30;

fs = 1000;                    % Sampling frequency (samples per second)
dt = 1/fs;                   % seconds per sample
StopTime = 4;                % seconds
t = (0:dt:StopTime)';        % seconds
F = 1;                       % Sine wave frequency (hertz)
r = 5*ones(length(t),1);           % Reference


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
N = 50; %prediction horizon
k_sim = length(r)-N;
Phi = []; Y = [];
%% recompute Theta 
for i = 1:length(y_data)-N;
if i == 1
y_ini = [0;0;0;0; y_data(i)];
u_ini = [0;0;0;0];
end
if i == 2 
y_ini = [0;0;0;y_data(i-1); y_data(i)];
u_ini = [0;0;0;u_data(i-1)];
end
if i == 3 
y_ini = [0;0;y_data(i-2);y_data(i-1); y_data(i)];
u_ini = [0;0;u_data(i-2);u_data(i-1)];
end
if i == 4 
y_ini = [0;y_data(i-3);y_data(i-2);y_data(i-1); y_data(i)];
u_ini = [0;u_data(i-3);u_data(i-2);u_data(i-1)];
end
if i >= 5 
y_ini = [y_data(i-4);y_data(i-3);y_data(i-2);y_data(i-1); y_data(i)];
u_ini = [u_data(i-4);u_data(i-3);u_data(i-2);u_data(i-1)];
end
uf = u_data(i:i+9)';

Phi = [Phi rbf_before_out_layer(centers_sigmas,u_ini,uf,y_ini)];
out_vector = [y_data(i+1);y_data(i+2);y_data(i+3);y_data(i+4);y_data(i+5);y_data(i+6);y_data(i+7);y_data(i+8);y_data(i+9);y_data(i+10)];
Y = [Y out_vector];
end

Theta = Y*pinv(Phi);

P    = Theta(:,1:50);
Gamma = Theta(:,31:end);


%% Build YALMIP SPC problem
Tini = 5;

centers = double(centers_sigmas.centers);
log_sigmas = double(centers_sigmas.log_sigmas);

Q = 200; 
R=  1;

Psi = kron(eye(N), R);
Omega = kron(eye(N), Q);

u = sdpvar(N,1);
y = sdpvar(N,1);
ref = sdpvar(N,1);
%u_ini = sdpvar(Tini-1, 1);
%y_ini = sdpvar(Tini, 1);
gvar = sdpvar(max(size(Phi*Phi')),1);
nl_part = sdpvar(length(centers),1);

objective = (y-ref)'*Omega*(y-ref)+(u)'*Psi*(u);

constraints = [y == Theta*[nl_part;u]];
for k = 1:N
    constraints = [constraints,  -10<=u(k)<=10];
end
Parameters = {nl_part, ref};
Outputs = {u};

options = sdpsettings('solver', 'quadprog', 'verbose', 0, 'debug', 0);

%options = sdpsettings('solver', 'quadprog', 'verbose', 0, 'debug', 0, 'osqp.eps_abs', 1e-3, 'osqp.eps_rel', 1e-3);
controller = optimizer(constraints, objective, options, Parameters, Outputs);

%% initial conditions
y(1) = 1;
xdot = [0;0];
x_sim = [0;0]';
u_mpc = 0;
u = [];
th=[];

%%
for i = 1:k_sim;
i
tic;
%%% u_ini & y_ini   
if i == 1
y_ini = [0;0;0;0; y(i)];
u_ini = [0;0;0;0];
end
if i == 2 
y_ini = [0;0;0;y(i-1); y(i)];
u_ini = [0;0;0;u(i-1)];
end
if i == 3 
y_ini = [0;0;y(i-2);y(i-1); y(i)];
u_ini = [0;0;u(i-2);u(i-1)];
end
if i == 4 
y_ini = [0;y(i-3);y(i-2);y(i-1); y(i)];
u_ini = [0;u(i-3);u(i-2);u(i-1)];
end
if i >= 5 
y_ini = [y(i-4);y(i-3);y(i-2);y(i-1); y(i)];
u_ini = [u(i-4);u(i-3);u(i-2);u(i-1)];
end

clear nl_part;
nl_part = [];
for im = 1:length(centers)
out = [u_ini;y_ini] - centers(im,:)';  
out = sqrtm(sum(out.^2))./ exp(log_sigmas(im));
out = exp(-1*out^2);
nl_part= [nl_part;out];
end


%%% mpc sol
[Uk , err] = controller({nl_part, r(i+1:i+N)});
u_mpc = Uk(1);
u = [u u_mpc];
th=[th;toc];

%State Derivative
mu = 1;
xdot(i+1,1) = x_sim(i,2);
xdot(i+1,2) = mu*(1-x_sim(i,1)^2)*x_sim(i,2)-x_sim(i,1)+u_mpc;

%State Update (Eulers Method)
x_sim(i+1,1) = x_sim(i)+dt*xdot(i,1);
x_sim(i+1,2) = x_sim(i)+dt*xdot(i,2);
y(i+1) = x_sim(i+1,1)
end

e = y-r(1:length(y));
%% Plots
figure;
plot(r(1:length(y)),'LineWidth',3);
hold on;
plot(y,'LineWidth',3);
legend('reference','RBF-SPC');
title ('Reference trajectory vs CL-MPC');
grid on

figure;
plot(u,'LineWidth',3);
title ('input');
grid on;
xlabel('Iterations');



figure;
plot(e,'LineWidth',3);
title ('error');
grid on;
xlabel('Iterations');

