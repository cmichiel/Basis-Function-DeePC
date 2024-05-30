%% N-step ahead predictor
clc; close all; clear all;

M = 0.5   ;       % mass of the pendulum
m = 0.2  ;       % Mass of the cart
L = 0.3   ;       % lenght of the pendulum
b = 0.1  ;       % friction coefficient
g = 9.81  ;      % acceleration of gravity


fs = 100;   % Sampling frequency (samples per second)
dt = 1/fs;  


%                  % seconds per sample
% StopTime = 100;                % seconds
% time = (0:dt:StopTime)';        % seconds
% F = 1;                       % Sine wave frequency (hertz)
% r = sin(2*pi*F*time);           % Reference

Range = [-1, 1];
SineData = [25, 40, 1];
Band = [0, 1];
NumPeriod = 1;
Period = 10001;
Nu = 1;

u_data = idinput([Period 1 NumPeriod],'sine',Band,Range,SineData)';
%u_data = idinput([Period 1 NumPeriod],'prbs',Band,Range,SineData)';
u_ID = iddata([],u_data,dt);

time = (0:length(u_data)-1) * dt;

%%
x_initial = [0,0,0,0];
%[y_data,x] = InvertCartPendulum( x_initial,u_data);
u_func = @(t) interp1(time', u_data, t);

x0 =  [x_initial(1);
    x_initial(2);
    x_initial(3);
    x_initial(4)]


y1 = [];
y2 = [];

[t,x] = ode45(@(t,x) ICP(t,x,u_func), time, x0);

%y = [x(1)';x(2)];






figure()
subplot(3,1,1)
plot(u_data)
title("Input data")
subplot(3,1,2)
plot(x(:,1))
title("Output data")
subplot(3,1,3)
plot(x(:,2))
title("Output data")
% 
% 
% u_data = u_data.'
% y_data = y_data.'
% save('u_data',"u_data")
% save('y_data',"y_data")



function dxdt = ICP(t,x,u_data)
dxdt = zeros(4,1);
M = 1.5   ;       % mass of the pendulum
m = 0.2  ;       % Mass of the cart
L = 0.3   ;       % lenght of the pendulum
b = 0.1  ;       % friction coefficient
g = 9.81  ;      % acceleration of gravity

dxdt(1) = x(3);
dxdt(2) = x(4);
dxdt(3) = (-m*L*sin(x(2))*x(4)^2  +m*g*cos(x(2))+  u_data(t)   )  /  (M+m*sin(x(2))^2);
dxdt(4) = (-m*L*cos(x(2))*sin(x(2))*x(4)^2  + u_data(t)*cos(x(2))  +  (M+m)*g*sin(x(2)) )  /  (L*(M+m*(1-cos(x(2))^2)));
end