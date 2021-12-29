function logMar = poiLogMarg(Yreg, Xreg, offset)

% to debug
% Yreg = Y(ii,:)';
% Xreg = THETA{g}(cc).X';
% offset = THETA{g}(cc).d';

% way 1: do gamma approximation to lambda
T = length(Yreg);
logMar = 0;

XX = diag(Xreg*Xreg');
avec = XX.^(-1);
bvec = XX.*exp(offset);
probvec = 1./(1 + bvec);
logMar =nansum(log(nbinpdf(Yreg, avec, probvec)));

end