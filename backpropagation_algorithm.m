
function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end

function g = sigmoidGradient(z)
    g = sigmoid(z) .* (1 - sigmoid(z));
end


inputneuron = 3;
hiddenlayerneuron = 5;
outputlayerneuron = 1;

w1 = 0.5 * ones(hiddenlayerneuron, inputneuron + 1);
w2 = -0.5 * ones(outputlayerneuron, hiddenlayerneuron + 1);

%data
x = [1 0 1; 0 1 1; 1 1 1];
y = [1; 0; 1];


alpha = 0.1;     % Learning rate
epochs = 1000;

% Training loop
for i = 1:epochs
    % Forward pass
    a1 = [ones(size(x, 1), 1), x];
    z2 = a1 * w1';
    a2 = [ones(size(z2, 1), 1), sigmoid(z2)];
    z3 = a2 * w2';
    a3 = sigmoid(z3);

    % Error calculation
    error = a3 - y;

    % Backward pass
    delta2 = error .* sigmoidGradient(z3);
    delta1 = (delta2 * w2(:, 2:end)) .* sigmoidGradient(z2);

    % Update weights
    w1 = w1 - alpha * (delta1' * a1);
    w2 = w2 - alpha * (delta2' * a2);
end
