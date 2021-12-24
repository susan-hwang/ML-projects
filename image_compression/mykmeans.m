function [ class, centroid ] = mykmeans( pixels, K )
%
% Your goal of this assignment is implementing your own K-means.
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
        clusterid = cluster_assign(pixels, centroid);
        
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

    % random initialize
    % centroid = randi([0, 255], K, d);
    % indx = randsample(n, K);
    % centroid = pixels(indx, :);
       
    clusterind = zeros(n, K); % use one-hot representation for computational convienence 
    idx = sub2ind(size(clusterind), 1:n, clusterid');
    clusterind(idx) = 1; 
    reconstruct_loss = sum(sum((pixels - clusterind * centroid).^2));
    previous_loss = 1e6;
    
    stop_threshold = 1e2;
    max_iter = 500;
    iter = 0;
    
    while abs(reconstruct_loss - previous_loss) > stop_threshold && iter < max_iter
        previous_loss = reconstruct_loss;
        
        % recompute centroid
        centroid = clusterind' * pixels;
        ncluster = sum(clusterind, 1);
        centroid = bsxfun(@times, centroid, 1./ncluster');
        
        % reassign clusterid
        clusterid = cluster_assign(pixels, centroid);
        idx = sub2ind(size(clusterind), 1:n, clusterid');
        clusterind = zeros(n, K);
        clusterind(idx) = 1; 
        
        % computer objective
        reconstruct_loss = sum(sum((pixels - clusterind * centroid).^2));
        
        iter = iter + 1;
    end
    
    [class, ~] = find(clusterind'~=0);
    
end


function clusterid = cluster_assign(x, center)
    xcdist = pdist2(x, center);
    [~, clusterid] = min(xcdist, [], 2);
end