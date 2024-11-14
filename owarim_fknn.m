function [accuracy, y_predicted] = owarim_fknn(xtrain, ytrain, xtest, ytest, k_values, alpha)

% =========================================================================
% Ordered weighted averaging with regular increasing monotone-based fuzzy
% k-nearest neighbor (OWARIM-FKNN) classifier

% INPUTS:
%   xtrain: train data
%   ytrain: train classes
%   xtest:  test data
%   ytest:  test classes
%   k_values: number of nearest neighbors
%   alpha: parameter value of thr RIM quantifier

% OUTPUTs:
% accuracy: classification accuracy over test data set
% y_predicted: predicted class labels

% 'RIM.m' and 'owamatrix.m' are required. 
% These files are required to compute RIM quantifier and OWA-based class prototypes

% Created by Mahinda Mailagaha Kumbure & Pasi Luukka, 11/2024 

% ==================================================================================

% start

% initialization
num_test = size(xtest,1); % # of test samples
m        = 2;             % fuzzy strength value


% for each test point, do:

for i=1:num_test
    
    clas_index = unique(ytrain);  % class labels in the train data
        
    for ii=1:length(clas_index)   % go through each class
    train_data_class_ii = xtrain(ytrain==clas_index(ii),:); % class subset
    num_train_ii        = size(train_data_class_ii,1);      % size of the class subset
     
 % (1) compute Euclidean distance between a test sample and training samples in each class
    distances = (repmat(xtest(i,:), num_train_ii,1) - train_data_class_ii).^2;
    distances = sum(distances,2)';
    
    [~, indices] = sort(distances); % sort the distance values 
    
    % find the indexes of the nearest neighbors
    if (num_train_ii<k_values)
    neighbor_index = indices;  
    k_values = num_train_ii;
    else
    neighbor_index = indices(1:k_values);
    end
    
    nn_set = train_data_class_ii(neighbor_index,:); % the set of k nearest neighbors
        
 % (2) calculate the RIM-based multi-local OWA vectors from each class
    local_mean    = zeros(size(nn_set));
    
    for jj=1:length(neighbor_index)
        if jj==1
        local_mean(jj,:) = nn_set(1:jj,:);
        else
        w = RIM(jj, alpha); % weights from RIM quantifier

        datas = nn_set(1:jj,:);
        local_mean(jj,:) = owamatrix(datas',w);
        end
    end
    
 % (3) calculate the distances from the test sample to k multi-local OWA vectors
    distances2   = (repmat(xtest(i,:), size(local_mean,1), 1) - local_mean).^2;
    distances2   = sum(distances2,2);
    
       % and define the weights for each multi-local OWA vector
            w2   = RIM(length(neighbor_index), alpha);
    
 % (4) compute the distance between the test sample and pseudo nearest neighbor from each class

    d_y_Xpnn(ii) = w2*distances2;
    labels3(ii)  = clas_index(ii); % class labels of pseudo nearest neighbors

    end
    
 % (5) compute fuzzy memberships for each class for test sample using the
 % distances between test sample and pseudo nearest neighbors 
 
    weight       = d_y_Xpnn.^(-1/(m-1)); % fuzzy weights
 
 	   % set the Inf (infite) weights, if there are any, to  1.
        if max(isinf(weight))
           weight(isinf(weight)) = 1;
        end
    
    % crisp method for defining u_ij
    labels_iter = zeros(length(labels3),max(ytrain));

    for jj=1:length(clas_index)
        labels_iter(jj,:) = [zeros(1, labels3(jj)-1) 1 zeros(1,max(ytrain) - labels3(jj))];
    end    
    
    % compute class memberships
	test_out = weight*labels_iter/(sum(weight));
    
 % (6) classify test sample into the class that has the highest membership
    [~, index_of_max]   = max(test_out');
    y_predicted(i)      = index_of_max;
    

end

accuracy = sum(y_predicted' == ytest)/length(ytest); 


