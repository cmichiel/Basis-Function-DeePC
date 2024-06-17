%% N-step ahead predictor
clc; close all; clear all;


fs = 10;                    % Sampling frequency (samples per second)
dt = 1/fs;                   % seconds per sample
% StopTime = 4;                % seconds
% t = (0:dt:StopTime)';        % seconds
% F = 1;                       % Sine wave frequency (hertz)
% r = sin(2*pi*F*t);           % Reference




%                  % seconds per sample
% StopTime = 100;                % seconds
% time = (0:dt:StopTime)';        % seconds
% F = 1;                       % Sine wave frequency (hertz)
% r = sin(2*pi*F*time);           % Reference

Range = [-10, 10];
SineData = [10,40, 1];
Band = [0, 1];
NumPeriod = 1;
Period = 10000;
Nu = 1;

u_data = idinput([Period 1 NumPeriod],'sine',Band,Range,SineData)';

%u_data = idinput([Period 1 NumPeriod],'prbs',Band,Range,SineData)';
u_ID = iddata([],u_data,dt);

time = (0:length(u_data)-1) * dt;

%%
x_initial = [1,1];
%[y_data,x] = InvertCartPendulum( x_initial,u_data);
u_func1 = @(t) interp1(time', u_data, t);



x0 =  [x_initial(1);
    x_initial(2)];



y1 = [];
y2 = [];

[t,x] = ode45(@(t,x) VDP(t,x,u_func1), time, x0);

%y = [x(1)';x(2)];

figure()
subplot(2,1,1)
title("VDP System")
plot(time,u_data)
ylabel("u_1")
xlabel("time in [s]")
subplot(2,1,2)
plot(time,x(:,1))
ylabel('x_1')
xlabel("time in [s]")



%u_data = [u_data1;u_data2].'
y_data = [x(:,1)].'
save('u_data10000',"u_data")
save('y_data10000',"y_data")



function dxdt = VDP(t,x,u_data)
dxdt = zeros(2,1);
%Model Parameters
mu = 1;

dxdt(1) = x(2);
dxdt(2) = mu*(1-x(1)^2)*x(2)-x(1)+u_data(t);
end