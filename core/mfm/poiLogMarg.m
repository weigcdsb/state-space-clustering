function logMar = poiLogMarg(Yreg, Xreg, offset)

% to debug
% Yreg = Y(ii,:)';
% Xreg = THETA{g}(c_prop).X';
% offset = THETA{g}(c_prop).d';

% way 1: do gamma approximation to lambda
T = length(Yreg);
% XX = diag(Xreg*Xreg');
% for kk = 1:T
%     f = @(lam) poisspdf(Yreg(kk), lam).*lognpdf(lam,offset(kk),XX(kk));
%     logMarvec(kk) = log(integral(f,0,Inf));
% end

XX = diag(Xreg*Xreg');
avec = XX.^(-1);
bvec = XX.*exp(offset);
probvec = 1./(1 + bvec);
logMarvec = log(nbinpdf(Yreg, avec, probvec));
logMar =nansum(logMarvec);

% way 2: Laplace approximation...




end