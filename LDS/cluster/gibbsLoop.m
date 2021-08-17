function [] = gibbsLoop(Y, Lab, d_tmp, C_tmp,... % cluster-invariant
    x0_tmp, Q0_tmp, b_tmp, A_tmp, Q_tmp, s_star,... % cluster-related
    mudc0, Sigdc0, mubA0_all, SigbA0_f, Psi0) % priors

% for development & debug...
% Lab = Z_fit(:,1);
% d_tmp = d_fit(:,1);
% C_tmp = C_fit(:,:,1);
% x0_tmp = x0_fit(:,1);
% Q0_tmp = Q0_fit(:,:,1);
% b_tmp = b_fit(:,1);
% A_tmp = A_fit(:,:,1);
% Q_tmp = Q_fit(:,:,1);

% output
N = size(Y, 1);
T = size(Y, 2);
p = length(x0_tmp)/N;

Xout = zeros(N*p, T);
dOut = zeros(N, 1);
COut = zeros(N, p);
x0Out = zeros(N*p, 1);
Q0Out = zeros(N*p, N*p);
bOut = zeros(N*p, 1);
AOut = zeros(N*p, N*p);
QOut = zeros(N*p, N*p);

[Zsort_tmp,idY] = sort(Lab);
uniZsort_tmp = unique(Zsort_tmp);
nClus_tmp = length(uniZsort_tmp);
latID = id2id(uniZsort_tmp, p);

C_trans_tmp = zeros(N, p*nClus_tmp);
for k = 1:nClus_tmp
    idx_tmp = (Zsort_tmp == uniZsort_tmp(k));
    C_trans_tmp(idx_tmp, ((k-1)*p+1):k*p) = C_tmp(idx_tmp,:);
end

% (1) update X_fit, x0_fit & Q0_fit
% adaptive smoothing
Y_tmp2 = Y(idY,:);
d_tmp2 = d_tmp(idY);
x0_tmp2 = x0_tmp(latID);
Q0_tmp2 = Q0_tmp(latID, latID);
A_tmp2 = A_tmp(latID, latID);
b_tmp2 = b_tmp(latID);
Q_tmp2 = Q_tmp(latID, latID);

[muX,Ws,~] = ppasmoo_poissexp_v2(Y_tmp2,C_trans_tmp,...
    d_tmp2,Q0_tmp2,A_tmp2,Q_tmp2);

hess_tmp = hessX(muX(:), d_tmp2, C_trans_tmp,...
    x0_tmp2, Q0_tmp2, Q_tmp2, A_tmp2, b_tmp2, Y_tmp2);

% use Cholesky decomposition to sample efficiently
R = chol(-hess_tmp,'lower'); % sparse
z = randn(length(muX(:)), 1) + R'*muX(:);
Xsamp = R'\z;

Xout(latID, :) = reshape(Xsamp,[], T);
x0Out(latID) = muX(:,1);
Q0Out(latID,latID) = Ws(:,:,1);

% (2) update d_fit & C_fit
% Laplace approximation
for i = 1:N
    latentId = id2id(Lab(i), p);
    X_tmpdc = [ones(1, T) ; Xout(latentId,:)]';
    lamdc = @(dc) exp(X_tmpdc*dc);
    
    derdc = @(dc) X_tmpdc'*(Y(i,:)' - lamdc(dc)) - Sigdc0\(dc - mudc0);
    hessdc = @(dc) -X_tmpdc'*diag(lamdc(dc))*X_tmpdc - inv(Sigdc0);
    [mudc,~,niSigdc,~] = newton(derdc,hessdc,...
        [d_tmp(i) C_tmp(i,:)]',1e-8,1000);
    
    Sigdc = -inv(niSigdc);
    Sigdc = (Sigdc + Sigdc')/2;
    dc = mvnrnd(mudc, Sigdc);
    dOut(i) = dc(1);
    COut(i,:) = dc(2:end);
end

% (3) update b_fit & A_fit
SigbA0 = SigbA0_f(nClus_tmp);
for l = uniZsort_tmp'
    latentId = id2id(l,p);
    mubA0 = mubA0_all(latentId, [1; latID+1]);
    mubA0 = mubA0(:);
    
    Xpost_tmp = Xout(latentId, 2:T);
    Xpost_tmp2 = Xpost_tmp(:);
    
    XbA_tmp = kron([ones(1,T-1); Xout(latID, 1:(T-1))]', eye(p));
    SigbA_tmp = inv(inv(SigbA0) +...
        XbA_tmp'*kron(eye(T-1), inv(Q_tmp(latentId,latentId)))*XbA_tmp);
    SigbA_tmp = sparse((SigbA_tmp + SigbA_tmp')/2);
    mubA_tmp = SigbA_tmp*(SigbA0\mubA0 +...
        XbA_tmp'*kron(eye(T-1), inv(Q_tmp(latentId,latentId)))*Xpost_tmp2);
    
    % TODO: chante to Cholesky decomposition later...
    bAtmp = reshape(mvnrnd(mubA_tmp, SigbA_tmp)', [], 1+nClus_tmp*p);
    bOut(latentId) = bAtmp(:,1);
    AOut(latentId, latID) = bAtmp(:,2:end);
end

% (4) update Q
for l = uniZsort_tmp'
    latentId = id2id(l,p);
    mux = AOut(latentId, latID)*Xout(latID, 1:(T-1)) + bOut(latentId);
    xq = Xout(latentId, 2:T) -mux;
    
    PsiQ = Psi0 + xq*xq';
    nuQ = T-1 + nu0;
    QOut(latentId,latentId) = iwishrnd(PsiQ,nuQ);
end


% labels without obs.: generate things by prior
outLab = setdiff(1:s_star, uniZsort_tmp);











end