function [XOut,x0Out,dOut,COut,bOut,AOut,QOut] =...
    blockDiag_gibbsLoop_MM(Y, Z_tmp, d_tmp, C_tmp,... % cluster-invariant
    x0_tmp, b_tmp, A_tmp, Q_tmp, kMM,... % cluster-related
    Q0, mux00, Sigx00, mudc0, Sigdc0, mubA0_mat, SigbA0_f, Psi0,nu0) % priors


% for development & debug...
% Z_tmp = Z_fit(:,g);
% d_tmp = d_fit(:,g-1);
% C_tmp = C_fit(:,:,g-1);
% x0_tmp = x0_fit(:,g-1);
% b_tmp = b_fit(:,g-1);
% A_tmp = A_fit(:,:,g-1);
% Q_tmp = Q_fit(:,:,g-1);

% output
N = size(Y, 1);
T = size(Y, 2);
p = length(x0_tmp)/kMM;

XOut = zeros(kMM*p, T);
x0Out = zeros(kMM*p, 1);
dOut = zeros(N, 1);
COut = zeros(N, p);
bOut = zeros(kMM*p, 1);
AOut = zeros(kMM*p, kMM*p);
QOut = zeros(kMM*p, kMM*p);

[Zsort_tmp,idY] = sort(Z_tmp);
uniZsort_tmp = unique(Zsort_tmp);
nClus_tmp = length(uniZsort_tmp);
latID = id2id(uniZsort_tmp, p);

% labels without obs.: generate things by prior
outLab = setdiff(1:kMM, uniZsort_tmp);
if(~isempty(outLab))
    
    x0Out =  mvnrnd(mux00, Sigx00)';
    for k =1:kMM
        latID_tmp = id2id(k, p);
%         mubA0_tmp = mubA0_mat(latID_tmp, :);
%         mubA0_tmp = mubA0_tmp(:);
%         
%         SigbA0_tmp = SigbA0_f(kMM);
%         
%         R = chol(inv(SigbA0_tmp),'lower');
%         z = randn(length(mubA0_tmp), 1) + R'*mubA0_tmp;
%         bASamp = R'\z;
%         bAtmp = reshape(bASamp, [], 1+kMM*p);
%         bOut(latID_tmp) = bAtmp(:,1);
%         AOut(latID_tmp, :) = bAtmp(:,2:end);
        
        QOut(latID_tmp,latID_tmp) = iwishrnd(Psi0,nu0);
    end
    
    bOut = zeros(kMM*p, 1);
    AOut = eye(kMM*p);
    
    invQ0 = inv(sparse(Q0));
    R = chol(invQ0,'lower');
    z = randn(kMM*p, 1) + R'*x0Out;
    XOut(:, 1) = R'\z;
end

% sort C and transform to match latents
Csort_trans_tmp = zeros(N, p*nClus_tmp);
for k = 1:nClus_tmp
    idx_old = (Z_tmp == uniZsort_tmp(k));
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


% (2) update x0_fit
invSigx0 = sparse(inv(Sigx00(latID, latID)) + inv(Q0_tmp2));
mux0 = invSigx0\(Sigx00(latID, latID)\mux00(latID) + Q0_tmp2\XOut(latID, 1));

R = chol(invSigx0,'lower'); % sparse
z = randn(length(mux0), 1) + R'*mux0;
x0Out(latID) = R'\z;

% (3) update d_fit & C_fit
% Laplace approximation

% for i = 1:N
%     latentId = id2id(Z_tmp(i), p);
%     X_tmpdc = [ones(1, T) ; XOut(latentId,:)]';
%     lamdc = @(dc) exp(X_tmpdc*dc);
%     
%     derdc = @(dc) X_tmpdc'*(Y(i,:)' - lamdc(dc)) - Sigdc0\(dc - mudc0);
%     hessdc = @(dc) -X_tmpdc'*diag(lamdc(dc))*X_tmpdc - inv(Sigdc0);
%     [mudc,~,niSigdc,~] = newton(derdc,hessdc,...
%         [d_tmp(i) C_tmp(i,:)]',1e-8,1000);
%     
%     Sigdc = -inv(niSigdc);
%     Sigdc = (Sigdc + Sigdc')/2;
%     dc = mvnrnd(mudc, Sigdc);
%     dOut(i) = dc(1);
%     COut(i,:) = dc(2:end);
% end

dOut = d_tmp;
COut = C_tmp;

% (4) update b_fit & A_fit
SigbA0 = SigbA0_f(nClus_tmp);
for l = uniZsort_tmp(:)'
    
    latentId = id2id(l,p);
    
    mubA0 = mubA0_mat(latentId, [1; latID+1]);
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

% (5) update Q
for l = uniZsort_tmp(:)'
    latentId = id2id(l,p);
    mux = AOut(latentId, latID)*XOut(latID, 1:(T-1)) + bOut(latentId);
    xq = XOut(latentId, 2:T) -mux;
    
    PsiQ = Psi0 + xq*xq';
    nuQ = T-1 + nu0;
    QOut(latentId,latentId) = iwishrnd(PsiQ,nuQ);
end

% labels without obs.: generate things by prior (remain XOut)
if(~isempty(outLab))
    outLatID = id2id(outLab , p);
    for t= 2:T
        XOut(outLatID, t) = mvnrnd(AOut(outLatID, :)*XOut(:, t-1) +...
            bOut(outLatID), QOut(outLatID, outLatID));
    end
end


end