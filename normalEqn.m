function [theta] = normalEqn(X, y)

%NORMALEQN Computes the closed-form solution to linear regression 

%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% Normal Equation = θ = (XTX)-1 * XT * y

theta = pinv(X' * X) * X' * y;

end
