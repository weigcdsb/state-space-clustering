function g = getGrad(f, x, step)

% 1. Set step size.
if ( nargin < 3 )
    step = eps(class(x))^(1/3);
end

% 2. Initialize output.
p = length(x);
g = zeros(p,1);

for i = 1:p
    % 3. Use central differences.
    x1    = x;
    x1(i) = x1(i) - step;
    
    x2    = x;
    x2(i) = x2(i) + step;
    g(i)      = (f(x2) - f(x1))/2/step;
end


% 4. Prevent Inf values in g.

% a. Ensure that g is a numeric, matrix and value is a numeric
% scalar.
assert( isnumeric(g) & ismatrix(g) );
assert( isnumeric(realmax) & isscalar(realmax) );

% b. Get abs(realmax).
absvalue = abs(realmax);

% c. Find +Inf or -Inf in g.
isinfg = isinf(g);

% d. Find +Inf/ -Inf elements in g and replace them with abs(realmax)/ -abs(realmax).
g(isinfg & g > 0) = absvalue;
g(isinfg & g < 0) = -absvalue;

end