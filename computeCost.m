function J = computeCostMulti(X, y, theta)

%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables

%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
% Cost = J(θ) = ½m Σ1m (hθ(xi) - yi)²

% Initialize some useful values
m = length(y); % number of training examples 
J = 0;

% Generate h(x) by multipying X (m X n+1) and Theta (n+1 X 1) => h (m X 1)
  h = X * theta;
  
% Next we find the error by doing predicted value - actual value 
  err = h - y;

% Based on the formula squaring the error value
sqrderr = err .^2;

% Calculating the sum of squared error
sse = sum(sqrderr);

% Cost is averaged over number of training examples
J = sse/(2*m);

end
