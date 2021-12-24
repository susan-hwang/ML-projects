function [ U, V , rec_err] = myRecommender( rateMatrix, lowRank )
   
    % Parameters
    maxIter = 500; % Choose your own.
    learningRate = 1e-3; % Choose your own.
    regularizer = 1e-3; % Choose your own.
    
    % Random initialization:
    [n1, n2] = size(rateMatrix);
    U = rand(n1, lowRank) / lowRank;
    V = rand(n2, lowRank) / lowRank;
    
    mask = (rateMatrix > 0);
    % Gradient Descent:
    % IMPLEMENT YOUR CODE HERE.
    pre_err = 1000;
    rec_err = [];
    for i = 1: maxIter
        temp = rateMatrix - U * V'; % n1-by-n2
        temp = temp .* mask;
        
        % update U and V
        Unext = (1 - learningRate * regularizer) * U + learningRate * temp * V;
        Vnext = (1 - learningRate * regularizer) * V + learningRate * temp' * U;
        
        U = Unext;
        V = Vnext;
        
        cur_err = norm((U*V' - rateMatrix) .* (rateMatrix > 0), 'fro') / sqrt(nnz(rateMatrix > 0)); 
        
        if abs(cur_err - pre_err) < 1e-6
            break
        end
        
        pre_err = cur_err;
        rec_err = [rec_err; cur_err];
        
    end
    
        
    
end