%AND GATE CODE USING PERCEPTRON NEURON

clc;
clear all;
x = [1 1;-1 -1;1 -1; -1 1];
t = [1;0;0;0];

% initialize weights and biases
R = 2; % number of inputs
N = 1;  % number of neurons
w = zeros(N,R);
b = 0; %bias
alpha = 1; %learning rate
theta = 0.5; %threshold

  for i=1:4
    yin = b+w(1)*x(i,1)+w(2)*x(i,2);
    if yin<theta
      y = 0;
    else
      y=1;
    endif
    if y-t(i) ~=0
     w(1) = w(1)+alpha*t(i)*x(i,1);
     w(2) = w(2)+alpha*t(i)*x(i,2);
     b = b+alpha*t(i);
    endif

endfor
disp(w(1));
w(2)
b
