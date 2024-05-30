function [y,x] = InvertCartPendulum(x_initial, u_data)
%Non-Linear State Space model of an Inverted Cart Pendulum model
%   Detailed explanation goes here


x0 =  [x_initial(1);
    x_initial(2);
    x_initial(3);
    x_initial(4)]


y1 = [];
y2 = [];

[t,x] = ode45(@(t,x) ICP(t,x,u_data), tspan, y0);

%y = [x(1)';x(2)];



function dxdt = ICP(t,x,u_data)
dxdt = zeros(4,1);
M = 1.5   ;       % mass of the pendulum
m = 0.2  ;       % Mass of the cart
L = 0.3   ;       % lenght of the pendulum
b = 0.1  ;       % friction coefficient
g = 9.81  ;      % acceleration of gravity

dxdt(1) = x(3);
dxdt(2) = x(4);
dxdt(3) = (-m*L*sin(x(2))*x(4)^2  +m*g*cos(x(2))+  u_data   )  /  (M+m*sin(x(2))^2);
dxdt(4) = (-m*L*cos(x(2))*sin(x(2))*x(4)^2  + u_data*cos(x(2))  +  (M+m)*g*sin(x(2)) )  /  ((M+m*(1-cos(x(2))^2))/L);
end
end