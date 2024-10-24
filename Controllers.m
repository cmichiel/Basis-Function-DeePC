clc,clear,close all
%%
%Author: Michiel Cevaal
%Date: 22-10-24

%%
%Sampling time and simulation time
fs = 10;                    % Sampling frequency (samples per second)
Ts = 1/fs;                   % seconds per sample
StopTime = 24;                % seconds
t = (0:Ts:StopTime-Ts)';        % seconds
%%
%Control Reference

%Stepped Reference
% A = [ 0.25:0.5:1.25,1.75,1.6:-0.75:-1.6,-1.75,-1.75:0.5:1.75,1.75,0,0,-1.75,-1.75,-0.25 ].*1.5;
% A = nonzeros(A)
% r = [];
% for j = 1:length(A)
% len = length(t)/length(A);
% r = [r, A(j)*ones(1,len)];
% end
% r = r;
% r = 1.5*ones(1,length(t));


%Sinusoidal Reference
F = 0.2; %Sinus frequency
A = 1; %Amplitude
r = A*sin(2*pi*F*t)';%Sinusoidal Reference

%%
%Select Model
N = 10; %Select Prediction Horizon
Tini =5; %Select Model Order

%Select Controller Type
% - Koopman
% - KoopmanResnet
% - NARX
% - NARXResnet
% - KoopmanNARX
BasisStructure = "KoopmanResnet"; 

%Select The Amount of Basis Functions
n_basisKoopman =80; %Select Basis Function Amount For Koopman Structure
n_basisNARX =40; % Select Basis Function Amount For NARX Structure 

% Select Basis Function type
% - gaussian
% - inverse multiquadratic
% - matern52
% - matern32

Basis_funcKoopman = 'gaussian'; %Select Basis Function Type for Koopman Structure
Basis_funcNARX = 'gaussian'; %Select Basis Function Type for NARX Structure


RBF = SelectRBF(N, Tini, BasisStructure, n_basisKoopman, n_basisNARX, Basis_funcKoopman, Basis_funcNARX)

%%
%Construct Phi and Theta

%Choose Input/Output Data
load('DataSets/u_data40sinsSweptAmpRed.mat')
load('DataSets/y_data40sinsSweptAmpRed.mat')

RBF.u_data = u_data;
RBF.y_data = y_data;
%Construct Phi, Theta and Yf
[RBF] = PhiTheta(RBF)

%%
%Construct YALMIP Control Problem

%Controller Type
% - SPC
% - DeePC-R1
% - DeePC-R2
% - DeePC-R3 (Currently not implemented)
Controller.Type = "SPC"

%Control Cost Parameters
Controller.Q = 100;
Controller.R = 0.5;
Controller.Rd = 0;
Controller.S = 100;
Controller.lambda=0;

[Controller] = SelectController(RBF, Controller)
%%
disp("Simulation for VDP system"+BasisStructure+"-"+Controller.Type+"Control")
%% initial conditions
y(1) = 1;
y_data = y(1);
iter = 1;
x0 = [1;1];
xk = x0;
xvec1 = [xk];
u_mpc = 0;
uvec = [0];
u = [];
th=[];
y_ini = ones(RBF.Tini,1)*y(1);
u_ini = zeros(RBF.Tini-1,1);
uf = zeros(RBF.N,1);

%%
figure;
plot(r(1:end-RBF.N),'LineWidth',3);
hold on;
yplot = plot(iter,y_data,'LineWidth',3);
legend('Reference','RBF-'+Controller.Type);
title (Controller.Type+': Reference vs Closed Loop Output');
grid on
%saveas(fig,'RBF-SPC\Figures\Closed Loop Output\VDP_NL-RBF-SPC_output_N'+string(N)+'_n_basis'+string(n_basis)+'_Tini'+string(Tini)+'_'+Basis_func+'.png')
yplot.YDataSource = 'y_data';
yplot.XDataSource = 'iter';

for i = 1:length(r)-RBF.N;
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


%%% DeePC Solution
if strcmp(RBF.BasisStructure, 'Koopman') || strcmp(RBF.BasisStructure, 'KoopmanResnet') || strcmp(RBF.BasisStructure, 'KoopmanNARX')
    clear nl_part;
    nl_part = [];
    nl_part = rbf_KoopmanPart(RBF,u_ini,y_ini);
    [UYK ] = Controller.controller({nl_part,u_mpc(1), r(i+1:i+RBF.N)',u_ini,y_ini});
else
    [UYK ] = Controller.controller({u_mpc(1), r(i+1:i+RBF.N)',u_ini,y_ini});
end
UYK = cell2mat(UYK);
Uk = UYK(:,1);
Yk(:,i) = UYK(:,2);
u_mpc = Uk(1);
uvec = [uvec u_mpc(1)];
th=[th;toc];

%State Derivative
mu = 1;
xk = [xk(1) + Ts*xk(2);  xk(2)+Ts*(mu*(1-xk(1)^2)*xk(2)-xk(1)+u_mpc(1))];
y(i+1) = xk(1);
y_data(i) = xk(1);
iter(i) = i;
xvec1 = [xvec1 xk];
refreshdata
drawnow
end
%%
%Compute Absolute Tracking Error 
Error = abs(y-r(1:length(y)));

%% Plots
fig = figure;
plot(r(1:length(y)),'LineWidth',3);
hold on;
plot(y,'LineWidth',3);
legend('Reference','RBF-'+ Controller.Type);
title (Controller.Type+' Control: Reference vs Closed Loop Output');
grid on
saveas(fig,'RBF-'+Controller.Type+'\Figures\Closed Loop Output\'+BasisStructure+'-'+Controller.Type+'\VDP_'+BasisStructure+'-RBF_STEPREF-'+Controller.Type+'_output'+RBF.ParameterString+'.png')


fig2 = figure;
plot(uvec,'LineWidth',3);
title(Controller.Type+': Control Input');
grid on
xlabel('Iterations');
saveas(fig2,'RBF-'+Controller.Type+'\Figures\Control Input\'+BasisStructure+'-'+Controller.Type+'\VDP_'+BasisStructure+'-RBF_STEPREF-'+Controller.Type+'_input'+RBF.ParameterString+'.png')


fig3 = figure;
plot(Error,'LineWidth',3);
title (Controller.Type+' Tracking Error');
grid on;
xlabel('Iterations');
saveas(fig3,'RBF-'+Controller.Type+'\Figures\Tracking Error\'+BasisStructure+'-'+Controller.Type+'\VDP_'+BasisStructure+'-RBF_STEPREF-'+Controller.Type+'_Error'+RBF.ParameterString+'.png')