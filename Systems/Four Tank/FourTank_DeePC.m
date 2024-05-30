%% N-step ahead predictor
clc; close all; clear all;

%Model Parameters
A1 = 50.27;
A2 = A1;
A3 = 28.27;
A4 = A3;
a1 = 0.233;
a2 = 0.242;
a3 = 0.127;
a4 = a3;
gamma1 = 0.4;
gamma2 = gamma1;
g = 981;

dt = 1.5;  
StopTime = 100;                % seconds
time = (0:dt:StopTime)';        % seconds


% use this for noiseless case
load('u_data.mat')
load('y_data.mat')

r = 6*ones(length(time),2);           % Reference
u_data = u_data(1,:);
y_data = y_data(1,:);
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
uf = u_data(i:i+(N-1))';

Phi = [Phi rbf_before_out_layer(centers_sigmas,u_ini,uf,y_ini)];
out_vector = [y_data(i+1);y_data(i+2);y_data(i+3);y_data(i+4);y_data(i+5);y_data(i+6);y_data(i+7);y_data(i+8);y_data(i+9);y_data(i+10)];
Y = [Y out_vector];
end

Theta = Y*pinv(Phi);

P    = Theta(:,1:30);
Gamma = Theta(:,31:end);

%% Build YALMIP SPC problem
Tini = 5;

centers = double(centers_sigmas.centers);
log_sigmas = double(centers_sigmas.log_sigmas);

Q = 200; 
R=  0.5;

Psi = kron(eye(N), R);
Omega = kron(eye(N), Q);

u = sdpvar(N,2);
y = sdpvar(N,2);
ref = sdpvar(N,2);
%u_ini = sdpvar(Tini-1, 1);
%y_ini = sdpvar(Tini, 1);
gvar = sdpvar(max(size(Phi*Phi')),2);
nl_part = sdpvar(length(centers),2);


objective = 0;
for p = 1:2
objective = objective + (y(:,p)-ref(:,p))'*Omega*(y(:,p)-ref(:,p))+(u(:,p))'*Psi*(u(:,p));
end


constraints = [y == Theta*[nl_part;u]];
for k = 1:N
    constraints = [constraints,  0<=u(k,1)<=20, 0<=u(k,2)<=20];
end
Parameters = {nl_part, ref};
Outputs = {u};

options = sdpsettings('solver', 'quadprog', 'verbose', 0, 'debug', 0);

%options = sdpsettings('solver', 'quadprog', 'verbose', 0, 'debug', 0, 'osqp.eps_abs', 1e-3, 'osqp.eps_rel', 1e-3);
controller = optimizer(constraints, objective, options, Parameters, Outputs);

%% initial conditions
y(1) = 10;
y(2) = 10;
x_sim = [10;10;10;10]';
xdot = [0;0;0;0];
x0 = x_sim;
u_mpc = [0;0];
u = [];
th=[];

%%
for i = 1:k_sim;
i
tic;
%%% u_ini & y_ini   
if i == 1
y_ini = [0,0;0,0;0,0;0,0; y(i,1),y(i,2)];
u_ini = [0,0; 0,0; 0,0; 0,0];
end
if i == 2 
y_ini = [0,0; 0,0;0,0 ;y(i-1,1), y(i-1,2); y(i,1),y(i,2)];
u_ini = [0,0; 0,0; 0,0; u(i-1,1),u(i-1,2)];
end
if i == 3 
y_ini = [0,0; 0,0;y(i-2,1), y(i-2,2) ;y(i-1,1), y(i-1,2); y(i,1),y(i,2)];    
u_ini = [0,0; 0,0;  u(i-2,1),u(i-2,2); u(i-1,1),u(i-1,2)]; 
end
if i == 4 
y_ini = [0,0; y(i-3,1), y(i-3,2);y(i-2,1), y(i-2,2) ;y(i-1,1), y(i-1,2); y(i,1),y(i,2)];     
u_ini = [0,0; u(i-3,1),u(i-3,2);  u(i-2,1),u(i-2,2); u(i-1,1),u(i-1,2)]; 
end
if i >= 5 
y_ini = [y(i-4,1), y(i-4,2); y(i-3,1), y(i-3,2);y(i-2,1), y(i-2,2) ;y(i-1,1), y(i-1,2); y(i,1),y(i,2)];  
u_ini = [u(i-4,1),u(i-4,2); u(i-3,1),u(i-3,2);  u(i-2,1),u(i-2,2); u(i-1,1),u(i-1,2)]; 
end

clear nl_part;
nl_part = [];
for im = 1:length(centers)
out1 = [u_ini(:,1);y_ini(:,1)] - centers(im,:)';  
out1 = sqrtm(sum(out1.^2))./ exp(log_sigmas(im));
out1 = exp(-1*out1^2);

out2 = [u_ini(:,2);y_ini(:,2)] - centers(im,:)';  
out2 = sqrtm(sum(out2.^2))./ exp(log_sigmas(im));
out2 = exp(-1*out2^2);
nl_part= [nl_part;out1,out2];
end


%%% mpc sol
[Uk, err ] = controller({nl_part, r(i+1:i+N,:)});
u_mpc = [Uk(1); Uk(11)]';
u = [u; u_mpc];
th=[th;toc];
%u_func1 = @(t) interp1(time(i:i+1)', u_mpc(:,1), t);
%u_func2 = @(t) interp1(time(i:i+1)', u_mpc(:,2), t);

% %%% State Derivative
xdot(i+1,1) = -a1/A1*sqrt(2*g*x_sim(i,1))+a3/A1*sqrt(2*g*x_sim(i,3))+ (gamma1 / A1) * u(i,1);
xdot(i+1,2) = -a2/A2*sqrt(2*g*x_sim(i,2))+a4/A1*sqrt(2*g*x_sim(i,4))+ (gamma2 / A2) *u(i,2);
xdot(i+1,3) = -a3/A3*sqrt(2*g*x_sim(i,3))+  ((1-gamma2)/ A3)   *u(i,2);
xdot(i+1,4) = -a4/A4*sqrt(2*g*x_sim(i,4))+  ((1-gamma1)/ A4)  *u(i,1);

%State Update (Eulers Method)
x_sim(i+1,1) = x_sim(i)+dt*xdot(i,1);
x_sim(i+1,2) = x_sim(i)+dt*xdot(i,2);
x_sim(i+1,3) = x_sim(i)+dt*xdot(i,3);
x_sim(i+1,4) = x_sim(i)+dt*xdot(i,4);
y(i+1,1) = x_sim(i+1,1);
y(i+1,2) = x_sim(i+1,2);
% xx2(i+1) = xx2(i)+Ts*xx1(i);
% y(i+1) = xx2(i+1);

%[t,x] = ode45(@FourTankSystem, [0 dt],x0,u(i:i+1,1),u(i:i+1,2));


end

e = y-r(1:length(y));




function dxdt = FourTankSystem(t, x , u_data1 , u_data2)
dxdt = zeros(4,1);
%Model Parameters
A1 = 50.27;
A2 = A1;
A3 = 28.27;
A4 = A3;
a1 = 0.233;
a2 = 0.242;
a3 = 0.127;
a4 = a3;
gamma1 = 0.4;
gamma2 = gamma1;
g = 981;

dxdt(1) = -a1/A1*sqrt(2*g*x(1))+a3/A1*sqrt(2*g*x(3))+ (gamma1 / A1) * u_data1;
dxdt(2) = -a2/A2*sqrt(2*g*x(2))+a4/A1*sqrt(2*g*x(4))+ (gamma2 / A2) * u_data2;
dxdt(3) = -a3/A3*sqrt(2*g*x(3))+  ((1-gamma2)/ A3)   *u_data2;
dxdt(4) = -a4/A4*sqrt(2*g*x(4))+  ((1-gamma1)/ A4)  *u_data1;
end