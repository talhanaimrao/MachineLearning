% McCulloch-Pitts Neuron

% Input vector (binary)
input_vector = [1 0 1 1];

% Weight
weights = [1 0 1 1];

% Threshold
threshold = 2;

% weighted sum
weighted_sum = sum(input_vector .* weights);

% Output activation
output = (weighted_sum >= threshold);

% Display results
disp('MP Neuron Output:');
disp(output);

