function [XOut,x0Out,dOut,COut,...
    mudcOut, SigdcOut,...
    bOut,AOut,QOut] =...
    full_gibbsLoop_DP_v2(Y,X_tmp, Z_tmp, d_tmp, C_tmp,...
    mudc_tmp, Sigdc_tmp,...
    x0_tmp, b_tmp, A_tmp, Q_tmp, s_star,...
    Q0_f, mux00_f, Sigx00_f, deltadc0, Taudc0,Psidc0,nudc0,...
    BA0_all_f, Lamb0_f, Psi0_f,nu0_f)

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

XOut = zeros(s_star*p, T);
x0Out = zeros(s_star*p, 1);
dOut = zeros(N, s_star);
COut = zeros(N, p*s_star);
mudcOut = zeros(p+1, s_star);
SigdcOut = zeros(p+1, p+1, s_star);
bOut = zeros(s_star*p, 1);
AOut = eye(s_star*p);
QOut = zeros(s_star*p, s_star*p);

[Zsort_tmp,idY] = sort(Z_tmp);
uniZsort_tmp = unique(Zsort_tmp);
nClus_tmp = length(uniZsort_tmp);
latID = id2id(uniZsort_tmp, p);

% labels without obs.: generate things by prior
outLab = setdiff(1:s_star, uniZsort_tmp);
if(~isempty(outLab))
    
    x0Out =  mvnrnd(mux00_f(s_star), Sigx00_f(s_star))';
    QOut = iwishrnd(Psi0_f(s_star), nu0_f(s_star));
    
    invQ0 = inv(sparse(Q0_f(s_star)));
    R = chol(invQ0,'lower');
    z = randn(s_star*p, 1) + R'*x0Out;
    XOut(:, 1) = R'\z;
end

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

% try
%     X_tmp2 = ppasmoo_poissexp_v2(Y_tmp2,Csort_trans_tmp,...
%         d_tmp2,x0_tmp2,Q0_tmp2,A_tmp2,b_tmp2,Q_tmp2);
% catch
%     disp('adaptive smoother failed, use previous step as initial')
%     X_tmp2 = X_tmp(latID, :);
% end
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
R = chol(-hess_tmp,'lower'); % sparse
z = randn(length(muXvec), 1) + R'*muXvec;
Xsamp = R'\z;
XOut(latID, :) = reshape(Xsamp,[], T);

% (2) update x0_fit
invSigx0 = sparse(inv(Sigx00_f(nClus_tmp)) + inv(Q0_tmp2));
mux0 = invSigx0\(Sigx00_f(nClus_tmp)\mux00_f(nClus_tmp) + Q0_tmp2\XOut(latID, 1));

R = chol(invSigx0,'lower'); % sparse
z = randn(length(mux0), 1) + R'*mux0;
x0Out(latID) = R'\z;

% (3) update d_fit & C_fit
for i = 1:N
    latentId = id2id(Z_tmp(i), p);
    X_tmpdc = [ones(1, T) ; XOut(latentId,:)]';
    lamdc = @(dc) exp(X_tmpdc*dc);
    
    derdc = @(dc) X_tmpdc'*(Y(i,:)' - lamdc(dc)) - Sigdc_tmp(:,:,Z_tmp(i))\(dc - mudc_tmp(:,Z_tmp(i)));
    hessdc = @(dc) -X_tmpdc'*diag(lamdc(dc))*X_tmpdc - inv(Sigdc_tmp(:,:,Z_tmp(i)));
    [mudc,~,niSigdc,~] = newton(derdc,hessdc,...
        [d_tmp(i, Z_tmp(i)) C_tmp(i,latentId)]',1e-6,1000);
%     [mudc,~,niSigdc,~] = newton(derdc,hessdc,...
%         deltadc0,1e-6,1000);
    if(sum(isnan(mudc)) ~= 0)
        [mudc,~,niSigdc,~] = newton(derdc,hessdc,...
        deltadc0,1e-6,1000);
    end
    
    R = chol(-niSigdc,'lower'); % sparse
    z = randn(length(mudc), 1) + R'*mudc;
    dc = R'\z;
    
    dOut(i,Z_tmp(i)) = dc(1);
    COut(i,latentId) = dc(2:end);
end


% (4) update mudc_fit & Sigdc_fit
dOut_new = zeros(N, s_star);
COut_new = zeros(N, p*s_star);

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



% (4)update Q
Y_BA = XOut(latID,2:T)';
X_BA = [ones(T-1,1) XOut(latID,1:(T-1))'];
Lamb0 = Lamb0_f(nClus_tmp);
BA0_all = BA0_all_f(s_star);

BA0 = BA0_all([1; latID+1],latID);
Psi0 = Psi0_f(nClus_tmp);
nu0 = nu0_f(nClus_tmp);

BAn = (X_BA'*X_BA + Lamb0)\(X_BA'*Y_BA + Lamb0*BA0);
PsiQ = Psi0 + (Y_BA - X_BA*BAn)'*(Y_BA - X_BA*BAn) +...
    (BAn - BA0)'*Lamb0*(BAn - BA0);
nuQ = T-1 + nu0;
QOut(latID,latID) = iwishrnd(PsiQ,nuQ);

% (5) update b_fit & A_fit
Lambn = X_BA'*X_BA + Lamb0;
BAvec = mvnrnd(BAn(:), kron(QOut(latID,latID), inv(Lambn)))';
BAsamp = reshape(BAvec,[], nClus_tmp*p)';
bOut(latID) = BAsamp(:,1);
AOut(latID, latID) = BAsamp(:,2:end);

% labels without obs.: generate things by prior (remain XOut)
if(~isempty(outLab))
    outLatID = id2id(outLab , p);
    
    for l=outLab(:)'
        
        outLatID_tmp = id2id(l,p);
        mudcOuter = mvnrnd(deltadc0, Taudc0);
        SigdcOuter = iwishrnd(Psidc0,nudc0);
        mudcOut(:,l) = mudcOuter;
        SigdcOut(:,:,l) = SigdcOuter;
        dcSampOut = mvnrnd(mudcOuter, SigdcOut(:,:,l), N);
        dOut(:,l) = dcSampOut(:,1);
        COut(:,outLatID_tmp) = dcSampOut(:,2:end);
%         x0Out(outLatID_tmp) = lsqr(COut(:,outLatID_tmp),...
%             (log(mean(Y(:,1:10),2))-dOut(:,l)));
%         XOut(outLatID_tmp, 1) = x0Out(outLatID_tmp);
%         XOut(outLatID_tmp, :) = ppasmoo_poissexp_v2(Y,COut(:,outLatID_tmp),dOut(:,l),...
%             x0Out(outLatID_tmp),Q0_f(1),...
%             AOut(outLatID_tmp, outLatID_tmp),...
%             bOut(outLatID_tmp),QOut(outLatID_tmp, outLatID_tmp));
        
    end
    
    for t= 2:T
        XOut(outLatID, t) = mvnrnd(AOut(outLatID, :)*XOut(:, t-1) +...
            bOut(outLatID), QOut(outLatID, outLatID));
    end
end



end