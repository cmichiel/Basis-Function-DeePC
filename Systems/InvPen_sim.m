%% N-step ahead predictor
clc; close all; clear all;

M = 0.5   ;       % mass of the pendulum
m = 0.2  ;       % Mass of the cart
L = 0.3   ;       % lenght of the pendulum
b = 0.1  ;       % friction coefficient
g = 9.81  ;      % acceleration of gravity


fs = 100;                    % Sampling frequency (samples per second)
dt = 1/fs;                   % seconds per sample
StopTime = 4;                % seconds
t = (0:dt:StopTime)';        % seconds
F = 1;                       % Sine wave frequency (hertz)
r = sin(2*pi*F*t);           % Reference

Range = [-10, 10];
SineData = [25, 40, 1];
Band = [0, 1];
NumPeriod = 1;
Period = 1000;
Nu = 1;

u_data = idinput([Period 1 NumPeriod],'sine',Band,Range,SineData)';
%u_data = idinput([Period 1 NumPeriod],'prbs',Band,Range,SineData)';
[y_data,x] = InvertCartPendulum(u_data);

figure()
subplot(2,1,1)
plot(u_data)
title("Input data")
subplot(2,1,2)
plot(y_data(2,:))
title("Output data")

u_data = u_data.'
y_data = y_data.'
save('u_data',"u_data")
save('y_data',"y_data")