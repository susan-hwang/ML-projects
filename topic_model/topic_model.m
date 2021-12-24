function [classind, mu, gamma] = topic_model( bow, K )
%
% Your goal of this assignment is implementing your own text clustering algo.
%
% Input:
%     bow: data set. Bag of words representation of text document as
%     described in the assignment.
%
%     K: the number of desired topics/clusters. 
%
% Output:
%     class: the assignment of each topic. The
%     assignment should be 1, 2, 3, etc. 
%
% For submission, you need to code your own implementation without using
% any existing libraries

% YOUR IMPLEMENTATION SHOULD START HERE!


[~, d] = size(bow);

% avoid numerical NAN or INF
bow = bow + 1e-12;

% initalize the model

p = ones(1, K) / K; % prior pi, we use uniform, can be changed to other
% mu = ones(d, K) / d;
mu = rand(d, K);
mu = bsxfun(@rdivide, mu, sum(mu, 1));

max_iter = 1000;

for iter = 1: max_iter
   
    % update gamma
    loggamma = bow * log(mu);
    loggamma = bsxfun(@plus, loggamma, log(p));

    % normalize 
    gamma = exp(loggamma);
    gamma = bsxfun(@rdivide, gamma, sum(gamma, 2));
    
    % update p
    p = sum(gamma, 1);
    p = bsxfun(@rdivide, p, sum(p));
    
    % update mu
    mu = bow' * gamma;
    mu = bsxfun(@rdivide, mu, sum(mu, 1));
    logmu = log(mu);
    
%     if sum(sum(isinf(logmu)))>0
%         % break
%         mu = mu + 1e-12;
%         mu = bsxfun(@rdivide, mu, sum(mu, 1));
%         % iter = 900;
%     end
    
end

[~, classind] = max(gamma, [], 2);

end

