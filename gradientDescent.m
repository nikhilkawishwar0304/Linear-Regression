function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

% Repeat Gradient Descent for pre-decided number of iterations
% Simultaneous update using of theta equation-
% theta(j) = theta(j) - alpha * derivative (J(theta)) with respect to theta(j)
% derivative = ∂J(θ)/∂(θj) = 1∕m Σ1m (hθx(i) - y(i)) * x(i)j
% hθ(x(i)) = θ0x0 + θ1x1 + θ2x2 + – – – + θnxn
 
for iter = 1:num_iters,

% Generate h(x) by multipying X (m X n+1) and Theta (n+1 X 1) => h (m X 1)
  h = X * theta;
  
% Next we find the error by doing predicted value - actual value 
  err = h - y;
  
% Finding the derivative term
  delt = X' * err;
  delt = delt/m;
  
% Simultaneous update using theta = theta - alpha * delta 
  theta = theta - (alpha * delt);


% Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
