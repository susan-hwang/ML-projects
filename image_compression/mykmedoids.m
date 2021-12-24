function [ class, centroid ] = mykmedoids( pixels, K )
%
% Your goal of this assignment is implementing your own K-medoids.
% Please refer to the instructions carefully, and we encourage you to
% consult with other resources about this algorithm on the web.
%
% Input:
%     pixels: data set. Each row contains one data point. For image
%     dataset, it contains 3 columns, each column corresponding to Red,
%     Green, and Blue component.
%
%     K: the number of desired clusters. Too high value of K may result in
%     empty cluster error. Then, you need to reduce it.
%
% Output:
%     class: the class assignment of each data point in pixels. The
%     assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
%     of class should be either 1, 2, 3, 4, or 5. The output should be a
%     column vector with size(pixels, 1) elements.
%
%     centroid: the location of K centroids in your result. With images,
%     each centroid corresponds to the representative color of each
%     cluster. The output should be a matrix with size(pixels, 1) rows and
%     3 columns. The range of values should be [0, 255].
%     
%
% You may run the following line, then you can see what should be done.
% For submission, you need to code your own implementation without using
% the kmeans matlab function directly. That is, you need to comment it out.

    % [class, centroid] = kmeans(pixels, K);

% We follow the wikipedia and use Partition Around Medoids
% algorithm to implement the Kmedoid with arbitrary distance. 

    distance = 'euclidean';
    % distance = 'cityblock';
    % distance = 'chebychev';
    % distance = 'mahalanobis';
    [n, d] = size(pixels); 
    
    % initialize the centroid: random or use PCA
    null_cluster = true;
    trials = 0;
    while null_cluster
        % random initialize
        % centroid = randi([0, 255], K, d);
        indx = randsample(n, K);
        centroid = pixels(indx, :);
        
        % assign clusters
        [clusterid, xcdist] = cluster_assign(pixels, centroid, distance);
        
        % check null clusters
        if length(unique(clusterid)) == K
            null_cluster = false;
        end
        trials = trials + 1;
        
        if trials >= 5
            K = K-1;
            fprintf('Too many clusters. Reduce to %d.\n', K);
            trials = 0;
        end
    end
    
    clusterind = zeros(n, K); % use one-hot representation for computational convienence 
    idx = sub2ind(size(clusterind), 1:n, clusterid');
    clusterind(idx) = 1; 

    previous_loss = 1e6;
    previous_cost = sum(xcdist .* clusterind, 1);
    reconstruct_loss = sum(previous_cost);
    new_cost = zeros(1, K);
    
    stop_threshold = 10;
    max_iter = 500;
    iter = 0;
    candi_pool = 100;
    
    while abs(reconstruct_loss - previous_loss) > stop_threshold && iter < max_iter
        previous_loss = reconstruct_loss;
        
        % recompute centroid for each cluster by swapping 
        for i = 1: K
            xi = pixels(clusterid == i, :);
            % exhaustive search requires to much memory
            % pd = pdist(xi, distance);
            candind = randsample(size(xi, 1), min(candi_pool, size(xi, 1)));
            candi = xi(candind, :);
            pd = pdist2(xi, candi, distance);
            total_dist = sum(pd, 1);
            [new_cost(i), indx] = min(total_dist);
            if new_cost(i) < previous_cost(i)
                previous_cost(i) = new_cost(i);
                centroid(i, :) = candi(indx, :);
            end
        end
        
        % reassign clusterid
        [clusterid, xcdist] = cluster_assign(pixels, centroid, distance);
        idx = sub2ind(size(clusterind), 1:n, clusterid');
        clusterind = zeros(n, K);
        clusterind(idx) = 1; 
        
        % computer objective
        reconstruct_loss = sum(sum(xcdist .* clusterind));
        
        iter = iter + 1;
    end
    
    [class, ~] = find(clusterind'~=0);

  
end

function [clusterid, xcdist] = cluster_assign(x, center, distance)
    xcdist = pdist2(x, center, distance);
    [~, clusterid] = min(xcdist, [], 2);
end