data = [1, 2, 3, 8, 9, 10];
grid_size = 3;
weights = [1.5, 5, 8];
learning_rate = 0.1;
epochs = 50;

for epoch = 1:epochs
    for i = 1:numel(data)
        distances = abs(data(i) - weights);
        [~, winner_neuron] = min(distances);
        for j = 1:numel(weights)
            weights(j) = weights(j) + learning_rate * (data(i) - weights(j));
        end
    end
end

figure;
scatter(data, zeros(size(data)), 30, 'filled');
hold on;
scatter(weights, zeros(size(weights)), 100, 'rx');
title('KSOM Clustering');
xlabel('Feature 1');
ylabel('Zero Line');
legend('Data Points', 'Neurons');
hold off;

