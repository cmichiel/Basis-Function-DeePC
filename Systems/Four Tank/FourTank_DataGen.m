%% N-step ahead predictor
clc; close all; clear all;


% fs = 100;                    % Sampling frequency (samples per second)
% dt = 1/fs;                   % seconds per sample
% StopTime = 4;                % seconds
% t = (0:dt:StopTime)';        % seconds
% F = 1;                       % Sine wave frequency (hertz)
% r = sin(2*pi*F*t);           % Reference

%fs = 100;   % Sampling frequency (samples per second)
dt = 1.5;  


%                  % seconds per sample
% StopTime = 100;                % seconds
% time = (0:dt:StopTime)';        % seconds
% F = 1;                       % Sine wave frequency (hertz)
% r = sin(2*pi*F*time);           % Reference

Range = [0, 60];
SineData = [25, 40, 1];
Band = [0, 1];
NumPeriod = 1;
Period = 1001;
Nu = 1;

u_data1 = idinput([Period 1 NumPeriod],'prbs',Band,Range,SineData)';
u_data2 = idinput([Period 1 NumPeriod],'prbs',Band,Range,SineData)';
u_data = [u_data1 ; u_data2];
%u_data = idinput([Period 1 NumPeriod],'prbs',Band,Range,SineData)';
u_ID = iddata([],u_data,dt);

time = (0:length(u_data)-1) * dt;

%%
x_initial = [10,10,10,10];
%[y_data,x] = InvertCartPendulum( x_initial,u_data);
u_func1 = @(t) interp1(time', u_data1, t);
u_func2 = @(t) interp1(time', u_data2, t);


x0 =  [x_initial(1);
    x_initial(2);
    x_initial(3);
    x_initial(4)]


y1 = [];
y2 = [];

[t,x] = ode45(@(t,x) FourTankSystem(t,x,u_func1,u_func2), time, x0);

%y = [x(1)';x(2)];

figure()
subplot(2,1,1)
plot(u_data(1,:))
title("Input data")
subplot(2,1,2)
plot(x(:,1))
title("Output data")

figure()
subplot(2,1,1)
plot(u_data(2,:))
title("Input data")
subplot(2,1,2)
plot(x(:,2))
title("Output data")


%u_data = [u_data1;u_data2].'
y_data = [x(:,1), x(:,2)].'
save('u_data',"u_data")
save('y_data',"y_data")



function dxdt = FourTankSystem(t,x,u_data1,u_data2)
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

dxdt(1) = -a1/A1*sqrt(2*g*x(1))+a3/A1*sqrt(2*g*x(3))+ (gamma1 / A1) * u_data1(t);
dxdt(2) = -a2/A2*sqrt(2*g*x(2))+a4/A1*sqrt(2*g*x(4))+ (gamma2 / A2) * u_data2(t);
dxdt(3) = -a3/A3*sqrt(2*g*x(3))+  ((1-gamma2)/ A3)   *u_data2(t);
dxdt(4) = -a4/A4*sqrt(2*g*x(4))+  ((1-gamma1)/ A4)  *u_data1(t);
end