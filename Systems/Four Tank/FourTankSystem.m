function [y,x] = FourTankSystem(x_initial, u_data)
%Non-linear State Space model of a 4 Tank System
%   Detailed explanation goes here

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

x1 =  x_initial(1);
x2 =  x_initial(2);
x3 =  x_initial(3);
x4 =  x_initial(4);

y1 = [];
y2 = [];

for i = 1:length(u_data(1,:))
i
%State updates
x1(i+1) = -a1/A1*sqrt(2*g*x1(i))+a3/A1*sqrt(2*g*x3(i))+gamma1/A1*u_data(1,i)
x2(i+1) = -a2/A2*sqrt(2*g*x2(i))+a4/A1*sqrt(2*g*x4(i))+gamma2/A2*u_data(2,i);
x3(i+1) = -a3/A3*sqrt(2*g*x3(i))+(1-gamma2)/A3*u_data(2,i);
x4(i+1) = -a4/A4*sqrt(2*g*x4(i))+(1-gamma1)/A4*u_data(1,i);

%Output updates
y1 = [y1;x1(i)];
y2 = [y2;x2(i)];
end

y = [y1';y2'];
x = [x1;x2;x3;x4];


end