function [XOut,x0Out,dOut,COut,bOut,AOut,QOut] =...
    blockDiag_gibbsLoop_DP(Y, Lab, d_tmp, C_tmp,... % cluster-invariant
    x0_tmp, b_tmp, A_tmp, Q_tmp, s_star,... % cluster-related
    Q0, mux00, Sigx00, mudc0, Sigdc0, mubA0_all, SigbA0_f, Psi0,nu0) % priors

% for development & debug...
% Lab = Z_fit(:,g-1);
% d_tmp = d_fit(:,g-1);
% C_tmp = C_fit(:,:,g-1);
% x0_tmp = x0_fit(:,g-1);
% b_tmp = b_fit(:,g-1);
% A_tmp = A_fit(:,:,g-1);
% Q_tmp = Q_fit(:,:,g-1);

% output
N = size(Y, 1);
T = size(Y, 2);
p = length(x0_tmp)/N;

XOut = zeros(N*p, T);
x0Out = zeros(N*p, 1);
dOut = zeros(N, 1);
COut = zeros(N, p);
bOut = zeros(N*p, 1);
AOut = zeros(N*p, N*p);
QOut = zeros(N*p, N*p);

[Zsort_tmp,idY] = sort(Lab);
uniZsort_tmp = unique(Zsort_tmp);
nClus_tmp = length(uniZsort_tmp);
latID = id2id(uniZsort_tmp, p);

% labels without obs.: generate things by prior
outLab = setdiff(1:s_star, uniZsort_tmp);
if(~isempty(outLab))
    fullLatid = id2id(1:s_star, p);
    
    x0Out(fullLatid) =  mvnrnd(mux00(fullLatid), Sigx00(fullLatid, fullLatid))';
    for k =1:s_star
        latID_tmp = id2id(k, p);
        mubA0_tmp = mubA0_all(latID_tmp, [1; fullLatid+1]);
        mubA0_tmp = mubA0_tmp(:);
        
        SigbA0_tmp = SigbA0_f(s_star);
        
        R = chol(inv(SigbA0_tmp),'lower');
        z = randn(length(mubA0_tmp), 1) + R'*mubA0_tmp;
        bASamp = R'\z;
        bAtmp = reshape(bASamp, [], 1+s_star*p);
        bOut(latID_tmp) = bAtmp(:,1);
        AOut(latID_tmp, fullLatid) = bAtmp(:,2:end);
        
        QOut(latID_tmp,latID_tmp) = iwishrnd(Psi0,nu0);
    end
    
    invQ0 = inv(sparse(Q0(fullLatid, fullLatid)));
    R = chol(invQ0,'lower');
    z = randn(s_star*p, 1) + R'*x0Out(fullLatid);
    XOut(fullLatid, 1) = R'\z;
end

% sort C and transform to match latents
Csort_trans_tmp = zeros(N, p*nClus_tmp);
for k = 1:nClus_tmp
    idx_old = (Lab == uniZsort_tmp(k));
    idx_sort = (Zsort_tmp == uniZsort_tmp(k));
    Csort_trans_tmp(idx_sort, ((k-1)*p+1):k*p) = C_tmp(idx_old,:);
end

% (1) update X_fit
% adaptive smoothing
Y_tmp2 = Y(idY,:);
d_tmp2 = d_tmp(idY);
x0_tmp2 = x0_tmp(latID);
Q0_tmp2 = Q0(latID, latID);
A_tmp2 = A_tmp(latID, latID);
b_tmp2 = b_tmp(latID);
Q_tmp2 = Q_tmp(latID, latID);

[muX,~,~] = ppasmoo_poissexp_v2(Y_tmp2,Csort_trans_tmp,...
    d_tmp2,x0_tmp2,Q0_tmp2,A_tmp2,b_tmp2,Q_tmp2);
hess_tmp = hessX(muX(:),d_tmp2,Csort_trans_tmp,Q0_tmp2,Q_tmp2,A_tmp2,Y_tmp2);

% use Cholesky decomposition to sample efficiently
R = chol(-hess_tmp,'lower'); % sparse
z = randn(length(muX(:)), 1) + R'*muX(:);
Xsamp = R'\z;
XOut(latID, :) = reshape(Xsamp,[], T);

% plot(XOut(latID, :)')
% plot(X')

% (2) update x0_fit
invSigx0 = sparse(inv(Sigx00(latID, latID)) + inv(Q0_tmp2));
mux0 = invSigx0\(Sigx00(latID, latID)\mux00(latID) + Q0_tmp2\XOut(latID, 1));

R = chol(invSigx0,'lower'); % sparse
z = randn(length(mux0), 1) + R'*mux0;
x0Out(latID) = R'\z;

% [x0Out(latID) x0']

% (3) update d_fit & C_fit
% Laplace approximation

for i = 1:N
    latentId = id2id(Lab(i), p);
    X_tmpdc = [ones(1, T) ; XOut(latentId,:)]';
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

% [dTest d(idPer)]
% [CTest C_all(idPer, :)]

% (4) update b_fit & A_fit
SigbA0 = SigbA0_f(nClus_tmp);
for l = uniZsort_tmp(:)'
    
    latentId = id2id(l,p);
    mubA0 = mubA0_all(latentId, [1; latID+1]);
    mubA0 = mubA0(:);
    
    Xpost_tmp = XOut(latentId, 2:T);
    Xpost_tmp2 = Xpost_tmp(:);
    
    XbA_tmp = kron([ones(1,T-1); XOut(latID, 1:(T-1))]', eye(p));
    invSigbA_tmp = sparse(inv(SigbA0) +...
        XbA_tmp'*kron(eye(T-1), inv(Q_tmp(latentId,latentId)))*XbA_tmp);
    
    mubA_tmp = invSigbA_tmp\(SigbA0\mubA0 +...
        XbA_tmp'*kron(eye(T-1), inv(Q_tmp(latentId,latentId)))*Xpost_tmp2);
    
    
    R = chol(invSigbA_tmp,'lower');
    z = randn(length(mubA_tmp), 1) + R'*mubA_tmp;
    bAtmp = reshape(R'\z, [], 1+nClus_tmp*p);
    
    bOut(latentId) = bAtmp(:,1);
    AOut(latentId, latID) = bAtmp(:,2:end);
end

% [bOut(latID) b]
% AOut(latID, latID)

% (5) update Q
for l = uniZsort_tmp(:)'
    latentId = id2id(l,p);
    mux = AOut(latentId, latID)*XOut(latID, 1:(T-1)) + bOut(latentId);
    xq = XOut(latentId, 2:T) -mux;
    
    PsiQ = Psi0 + xq*xq';
    nuQ = T-1 + nu0;
    QOut(latentId,latentId) = iwishrnd(PsiQ,nuQ);
end

% QOut(latID, latID)

% labels without obs.: generate things by prior (remain XOut)
if(~isempty(outLab))
    outLatID = id2id(outLab , p);
    for t= 2:T
        XOut(outLatID, t) = mvnrnd(AOut(outLatID, fullLatid)*XOut(fullLatid, t-1) +...
            bOut(outLatID), QOut(outLatID, outLatID));
    end
end





end