% Hebb Neuron
clear all
clc
% Input
x = [1 1; -1 1; 1 -1; -1 -1];

% Initialize weights
w = zeros(1, size(x, 2));

% Hebbian learning
for y = x'
    w = w + y' * y;
end

% Test the neuron with a new y
test_pattern = [1 -1];
output = sign(sum(w .* test_pattern));

% Display the final weights and output
disp('Final Weights:');
disp(w);
disp('Output for Test Pattern:');
disp(output);

