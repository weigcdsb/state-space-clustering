function logMar = poiLogMarg_PAL(Yreg, Xreg, offset, prior, cheb_a, cheb_b)

% to debug
% Yreg = Y(ii,:)';
% Xreg = THETA{g}(cc).X';
% offset = THETA{g}(cc).d';
% cheb_a = cheb_neuron_a(ii,:)';
% cheb_b = cheb_neuron_b(ii,:)';

invSig = 2*Xreg'*diag(cheb_a)*Xreg + inv(prior.SigC0);
Sig = inv(invSig);
mu = Sig*Xreg'*(Yreg - cheb_b - 2*diag(cheb_a)*offset);

logMar =offset'*(Yreg - cheb_b) - 0.5*log(det(prior.SigC0)) - offset'*diag(cheb_a)*offset +...
    0.5*log(det(Sig)) +  0.5*mu'*invSig*mu;




end