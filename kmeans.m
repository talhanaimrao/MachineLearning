data = [2.1, 2.3, 2.5, 7.8, 8.1, 8.5];
centroids = [2, 8];
k = numel(centroids);
N = numel(data);
labels = zeros(N, 1);

for i = 1:N

    distances = abs(data(i) - centroids);

       [~, clusterIndex] = min(distances);

    labels(i) = clusterIndex;
end

figure;
scatter(data, zeros(size(data)), 30, labels, 'filled');
hold on;
scatter(centroids, zeros(size(centroids)), 100, 'rx', 'LineWidth', 2);
title('Final State');
xlabel('Feature 1');
ylabel('Zero Line');
legend('Data Points', 'Centroids');
hold off;

