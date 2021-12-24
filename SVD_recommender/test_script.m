clear;

% Use real data:
load ('movie_data');
rateMatrix = train;
testMatrix = test;

% Global SVD Test:
lowRank = [1, 3, 5, 7, 10];
res = {}
for l=1:size(lowRank, 2)
    tic;
    [U, V] = myRecommender(rateMatrix, lowRank(l));
    % [U, V, rec_err] = myRecommender(rateMatrix, lowRank(l));
    % res{l} = rec_err;

    logTime = toc;
    
    trainRMSE = norm((U*V' - rateMatrix) .* (rateMatrix > 0), 'fro') / sqrt(nnz(rateMatrix > 0));
    testRMSE = norm((U*V' - testMatrix) .* (testMatrix > 0), 'fro') / sqrt(nnz(testMatrix > 0));
    
    fprintf('SVD-%d\t%.4f\t%.4f\t%.2f\n', lowRank(l), trainRMSE, testRMSE, logTime);
end

%%


color = ['c', 'r', 'b', 'g', 'm'];
for l = 1: length(lowRank)
    plot(1:length(res{l}), res{l}, 'r', 'Color', color(l), 'LineWidth', 3)
    hold on
end

legend('Rank-1', 'Rank-3', 'Rank-5', 'Rank-7', 'Rank-10')
