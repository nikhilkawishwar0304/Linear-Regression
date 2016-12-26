function [X_norm, mu, sigma] = featureNormalize(X)

%FEATURENORMALIZE Normalizes the features in X 

%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% Set the value of X to X-norm
X_norm = X;

% Returns a row vector of size = number of columns in X_norm 
% and fills it with 0
mu = zeros(1, size(X_norm, 2));
sigma = zeros(1, size(X_norm, 2));

% Returns a row vector with means of all the columns
mu = mean(X_norm);
% Returns a row vector with standard deviation of all the columns
sigma = std(X_norm);

% For all the columns mean normalise the matrix X
% x - mu / sd
for i = 1:(size(X_norm,2)),
  X_norm(:,i) = X_norm(:,i) - mu(i);
  X_norm(:,i) = X_norm(:,i) / sigma(i);
end;

% ============================================================

end
