function [XOut,x0Out,dOut,COut,...
    mudcOut, SigdcOut,...
    bOut,AOut,QOut, optdc, epsilon] =...
    update_all(Y,X_tmp, Z_tmp, d_tmp, C_tmp,...
    mudc_tmp, Sigdc_tmp,...
    x0_tmp, b_tmp, A_tmp, Q_tmp,...
    Q0_f, mux00_f, Sigx00_f, deltadc0, Taudc0,Psidc0,nudc0,...
    BA0, Lamb0, Psi0,nu0, g, optdc, epsilon, burnIn)

% for development & debug...
% Z_tmp = Z_fit(:,g-1);
% X_tmp = X_fit{g-1};
% d_tmp = d_fit{g-1};
% C_tmp = C_fit{g-1};
% mudc_tmp = mudc_fit{g-1};
% Sigdc_tmp = Sigdc_fit{g-1};
% x0_tmp = x0_fit{g-1};
% b_tmp = b_fit{g-1};
% A_tmp = A_fit{g-1};
% Q_tmp = Q_fit{g-1};

% output
N = size(Y, 1);
T = size(Y, 2);
p = 2;

XOut = X_tmp;
x0Out = x0_tmp;
dOut = d_tmp;
COut = C_tmp;
mudcOut = mudc_tmp;
SigdcOut = Sigdc_tmp;
bOut = b_tmp;
AOut = A_tmp;
QOut = Q_tmp;

[Zsort_tmp,idY] = sort(Z_tmp);
uniZsort_tmp = unique(Zsort_tmp);
nClus_tmp = length(uniZsort_tmp);
latID = id2id(uniZsort_tmp, p);

% sort C and transform to match latents
Csort_trans_tmp = zeros(N, p*nClus_tmp);
for k = 1:nClus_tmp
    latid_old = id2id(uniZsort_tmp(k), p);
    latid_new = id2id(k,p);
    idx_old = (Z_tmp == uniZsort_tmp(k));
    idx_sort = (Zsort_tmp == uniZsort_tmp(k));
    Csort_trans_tmp(idx_sort, latid_new) = C_tmp(idx_old,latid_old);
end

I = (1 : size(d_tmp, 1)) .';
k = sub2ind(size(d_tmp), I, Z_tmp);
d_new = d_tmp(k);

% (1) update X_fit
Y_tmp2 = Y(idY,:);
d_tmp2 = d_new(idY);
x0_tmp2 = x0_tmp(latID);
Q0_tmp2 = Q0_f(nClus_tmp);
A_tmp2 = A_tmp(latID, latID);
b_tmp2 = b_tmp(latID);
Q_tmp2 = Q_tmp(latID, latID);
X_tmp2 = X_tmp(latID, :);
gradHess = @(vecX) gradHessX(vecX, d_tmp2, Csort_trans_tmp, x0_tmp2, Q0_tmp2, Q_tmp2, A_tmp2, b_tmp2, Y_tmp2);
[muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp2(:),1e-6,1000);

if(sum(isnan(muXvec)) ~= 0)
    disp('use adaptive smoother initial')
    X_tmp2 = ppasmoo_poissexp_v2(Y_tmp2,Csort_trans_tmp,...
        d_tmp2,x0_tmp2,Q0_tmp2,A_tmp2,b_tmp2,Q_tmp2);
    [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp2(:),1e-6,1000);
end

% use Cholesky decomposition to sample efficiently
RX = chol(-hess_tmp,'lower'); % sparse
zX = randn(length(muXvec), 1) + RX'*muXvec;
Xsamp = RX'\zX;
XOut(latID, :) = reshape(Xsamp,[], T);
XOut(latID, :) = XOut(latID, :) - min(XOut(latID, :),[], 2);
XOut(latID, :) = (diag(range(XOut(latID, :),2)))\XOut(latID, :);

% (2) update x0_fit
invSigx0 = sparse(inv(Sigx00_f(nClus_tmp)) + inv(Q0_tmp2));
mux0 = invSigx0\(Sigx00_f(nClus_tmp)\mux00_f(nClus_tmp) + Q0_tmp2\XOut(latID, 1));

R = chol(invSigx0,'lower'); % sparse
z = randn(length(mux0), 1) + R'*mux0;
x0Out(latID) = R'\z;

% (3) update d_fit & C_fit
if(g < burnIn)
    tuneState = 1; % change epsilon
    disp("iter " + g + ", changing");
elseif(g == burnIn)
    tuneState = 2; % tune epsilon
    disp("iter " + g + ", tuning");
else
    tuneState = 3; % fix epsilon
    disp("iter " + g + ", tuned");
end

for i = 1:N
    l = Z_tmp(i);
    latentId = id2id(l, p);
    X_tmpdc = [ones(1, T) ; XOut(latentId,:)]';
    lamdc = @(dc) exp(X_tmpdc*dc);
    
    lpdf = @(dc) sum(log(poisspdf(Y(i,:)', lamdc(dc)))) +...
        log(mvnpdf(dc, mudc_tmp(:,l), Sigdc_tmp(:,:,l)));
    glpdf = @(dc) X_tmpdc'*(Y(i,:)' - lamdc(dc)) - Sigdc_tmp(:,:,l)\(dc - mudc_tmp(:,l));
    fg=@(dc_r) deal(lpdf(dc_r'), glpdf(dc_r')'); % log density and gradient
    dc0 = [d_tmp(i, l) C_tmp(i,latentId)]';
    
    switch tuneState
        case 1
            [dc_NUTS, ~, ~]=hmc_nuts(fg, dc0',optdc);
        case 2
            optdc.Madapt=50;
            [dc_NUTS, ~, diagn]=hmc_nuts(fg, dc0',optdc);
            epsilon(i) = diagn.opt.epsilonbar;
            optdc.Madapt=0;
        case 3
            optdc.epsilon = epsilon(i);
            [dc_NUTS, ~, ~]=hmc_nuts(fg, dc0',optdc);
    end
    
    dOut(i,l) = dc_NUTS(end,1);
    COut(i,latentId) = dc_NUTS(end,2:end);
end

% (4) update mudc_fit & Sigdc_fit
dOut_new = d_tmp;
COut_new = C_tmp;

for l = uniZsort_tmp(:)'
    dc_tmp = [dOut(Z_tmp == l, l) COut(Z_tmp == l, id2id(l,p))];
    invTaudc = inv(Taudc0) + sum(Z_tmp == l)*inv(Sigdc_tmp(:,:,l));
    deltadc = invTaudc\(Taudc0\deltadc0 + Sigdc_tmp(:,:,l)\sum(dc_tmp,1)');
    mudcOut(:,l) = mvnrnd(deltadc, inv(invTaudc));
    
    % different covariances
    dcRes = dc_tmp' - mudcOut(:,l);
    Psidc = Psidc0 + dcRes*dcRes';
    nudc = sum(Z_tmp == l) + nudc0;
    SigdcOut(:,:,l) = iwishrnd(Psidc,nudc);
    
    dc_samp = mvnrnd(mudcOut(:,l), SigdcOut(:,:,l), N);
    dOut_new(:,l) = dc_samp(:,1);
    dOut_new(Z_tmp == l, l) = dOut(Z_tmp == l, l);
    
    COut_new(:,id2id(l,p)) = dc_samp(:,2:end);
    COut_new(Z_tmp == l, id2id(l,p)) = COut(Z_tmp == l, id2id(l,p));
    
end

dOut = dOut_new;
COut = COut_new;

for l = uniZsort_tmp(:)'
    latentId = id2id(l,p);
    for k = latentId(:)'
        Y_BA = XOut(k,2:T)';
        X_BA = [ones(T-1,1) XOut(k,1:(T-1))'];
        
        BAn = (X_BA'*X_BA + Lamb0)\(X_BA'*Y_BA + Lamb0*BA0);
        PsiQ = Psi0 + (Y_BA - X_BA*BAn)'*(Y_BA - X_BA*BAn) +...
            (BAn - BA0)'*Lamb0*(BAn - BA0);
        nuQ = T-1 + nu0;
        QOut(k,k) = iwishrnd(PsiQ,nuQ);
        
        % (6) update b_fit & A_fit
        Lambn = X_BA'*X_BA + Lamb0;
        BAsamp = mvnrnd(BAn(:), kron(QOut(k,k), inv(Lambn)))';
        bOut(k) = BAsamp(1);
        AOut(k,k) = BAsamp(2);
    end
end

end