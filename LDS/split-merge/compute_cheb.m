function what_cheby = compute_cheb(xlim)

% do debug
% xlim = log(mean(Y(1,:))) + [-2 2];

dx = 0.01;
power = 2;

xx = (xlim(1) + dx/2):dx:xlim(2);
nx = length(xx);
xxw = (-1+1/nx):(1/(0.5*nx)):1;

Bx = zeros(nx, power + 1);

for ii = 1:(power + 1)
    Bx(:,ii) = xx.^(ii-1);
end

errwts_cheby = 1./sqrt(1-xxw.^2);
Dx = diag(errwts_cheby);

fx = exp(xx);
what_cheby = lsqr(Bx'*Dx*Bx, Bx'*Dx*fx');



end