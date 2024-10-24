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

Range = [-1, 1];
SineData = [80,10, 1];
Band = [0, 1];
NumPeriod = 1;
Period = 10000;
Nu = 1;


u_data_train = [];
Samples_train = 5120;
for i = 1:4
Range = [-i, i];
u_data_train = [u_data_train, idinput([Samples_train/4 1 NumPeriod],'sine',Band,Range,SineData)'];
end



u_data_test = [];
Samples_test = 2560;
for i = 1:4
Range = [-i, i];
u_data_test = [u_data_test, idinput([Samples_test/4 1 NumPeriod],'sine',Band,Range,SineData)'];
end
u_data = [u_data_train,u_data_test];


% u_data = idinput([Period 1 NumPeriod],'sine',Band,Range,SineData)';

%u_data = idinput([Period 1 NumPeriod],'prbs',Band,Range,SineData)';
u_ID = iddata([],u_data,dt);

time = (0:length(u_data)-1) * dt;

%%
x_initial = [1,1];
u_func1 = @(t) interp1(time', u_data, t);



x0 =  [x_initial(1);
    x_initial(2)];

mu = 1;


y_train(1) = x_initial(1);
xk = x0;


for i = 1:Samples_train
xk = [xk(1) + dt*xk(2);  xk(2)+dt*(mu*(1-xk(1)^2)*xk(2)-xk(1)+exp(u_data_train(i)))];
y_train(i) = xk(1);
end

x0 =  [x_initial(1);
    x_initial(2)];

mu = 1;


y_test(1) = x_initial(1);
xk = x0;

for i = 1:Samples_test
xk = [xk(1) + dt*xk(2);  xk(2)+dt*(mu*(1-xk(1)^2)*xk(2)-xk(1)+exp(u_data_test(i)))];
y_test(i) = xk(1);
end
y = [y_train, y_test]
%y = [x(1)';x(2)];

figure()
subplot(2,1,1)
title("VDP System")
plot(time,u_data)
ylabel("u_1")
xlabel("time in [s]")
subplot(2,1,2)
plot(y')
ylabel('x_1')
xlabel("time in [s]")



%u_data = [u_data1;u_data2].'
y_data = y;
%%
save('Datasets/u_data80sinsSweptAmpRedEXP.mat',"u_data")
save('Datasets/y_data80sinsSweptAmpRedEXP.mat',"y_data")

%%
% [coeff,score,latent,tsquared] = pca([u_data; y_data]')
% scatter(score(:,1),score(:,2))
% 
% xlabel('1st Principal Component')
% ylabel('2nd Principal Component')


% [Y,loss] = tsne([u_data; y_data]');
% figure;
% gscatter(Y(:,1),Y(:,2))