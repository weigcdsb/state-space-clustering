addpath(genpath('D:\github\state-space-clustering'));

%% simulation
rng(2)
n = 10;
nClus = 3;
p = 2;
T = 1000;

Lab = repelem(1:nClus, n);
d = ones(n*nClus,1)*0.5;
C_all = reshape(normrnd(1e-2,1e-3,n*nClus*p,1), [], p);
C_trans = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_all(k,:);
end

X1 = zeros(p, T);
X2 = zeros(p, T);
X3 = zeros(p, T);

Q0 = eye(p)*1e-3;
mu1 = [1 0.5]*20;
mu2 = [0.5 1]*15;
mu3 = [1 0.2]*10;
X1(:,1) = mvnrnd(mu1,Q0);
X2(:,1) = mvnrnd(mu2,Q0);
X3(:,1) = mvnrnd(mu3,Q0);
X = [X1; X2; X3];

b1 = ones(p,1)*0.05;
b2 = -ones(p,1)*0.05;
b3 = -ones(p,1)*0.05;
b = [b1;b2;b3];

Q1 = 1e-3*eye(p);
Q2 = 1e-4*eye(p);
Q3 = 1e-5*eye(p);
Q = blkdiag(Q1, Q2, Q3);

A_in = reshape(normrnd(0,1e-3,p*p*nClus,1), [], p, nClus);
A = blkdiag(A_in(:,:,1), A_in(:,:,2), A_in(:,:,3));
A((p+1):2*p, 1:p) = reshape(normrnd(1e-3,5*1e-4,p*p,1), [], p);
A(1:p, (p+1):2*p) = reshape(normrnd(3*1e-3,5*1e-4,p*p,1), [], p);
A((2*p+1):3*p, 1:p) = reshape(normrnd(1e-3,5*1e-4,p*p,1), [], p);
A(1:p, (2*p+1):3*p) = reshape(normrnd(3*1e-5,6*1e-4,p*p,1), [], p);
Aplot = A;
A(eye(size(A)) > 0) = 1;

figure(1)
imagesc(Aplot)
colorbar()
xlabel('sending')
ylabel('receiving')
% A = eye(nClus*p);

% let's generate lambda
logLam = zeros(n*nClus, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    X(:, t) = mvnrnd(A*X(:, t-1) + b, Q)';
    logLam(:, t) = d + C_trans*X(:,t);
end

figure(2)
imagesc(exp(logLam))
colorbar()

Y = poissrnd(exp(logLam));
figure(3)
imagesc(Y)
colorbar()

figure(4)
clusterPlot(Y, Lab)

figure(5)
plot(X')

plot(X(1:p,:)')
plot(X(p+1:2*p,:)')
plot(X(2*p+1:3*p,:)')



%% fitting
% assume now observe Y and Z
% later Y will be the only observation

ng = 100;

% pre-allocation
X_fit = zeros(nClus*p, T, ng);
d_fit = zeros(n*nClus, ng);
C_fit = zeros(n*nClus, p, ng);
x0_fit = zeros(nClus*p, ng);
Q0_fit = zeros(p, p, ng);
A_fit = zeros(nClus*p, nClus*p, ng);
b_fit = zeros(nClus*p, ng);
Q_fit = zeros(nClus*p, nClus*p, ng);

% initials
% initial for d_fit: 0
C_fit(:,:,1) = reshape(normrnd(0,1,n*nClus*p,1), [], p);
A_fit(:,:,1) = eye(nClus*p);
% initial for b_fit: 0
Q_fit(:,:,1) = eye(nClus*p)*1e-4;

C_trans_tmp = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans_tmp(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,1);
end

x0_fit(:,1) = lsqr(C_trans_tmp,(log(mean(Y(:,1:10),2))-d_fit(:,1)));
Q0_fit(:,:,1) = eye(p);
Qc = repmat({Q0_fit(:,:,1)}, 1, nClus);
Q0Filt = blkdiag(Qc{:});

[X_fit(:,:,1),~,~] = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_fit(:,1),...
    x0_fit(:,1),Q0Filt,A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));

% prior parameters
mud0 = zeros(n*nClus,1);
Sigd0 = eye(n*nClus);

muCj0 = zeros(n*p,1);
SigCj0 = eye(n*p);

mux00 = zeros(nClus*p,1);
Sigx00 = eye(nClus*p);

Psi00 = eye(p);
nu00 = p+2;

muAjl0 = eye(p);
muAjl0 = muAjl0(:);
SigAjl0 = eye(p*p);

mub0 = zeros(nClus*p, 1);
Sigb0 = eye(nClus*p);

Psi0 = eye(p)*1e-4;
nu0 = p+2;

% Please do all the things blockwise,...
% to transfer smoothly to clustering problem
for g = 2:10
    
    % (1) update latent vectors x_t^{(j)}: PASS
    % cannot do blockwise, because of interaction...
    % in the clustering case, need to reorder the observation and matrix...
    % here, I just ignore that for simpicity...
    Qc = repmat({Q0_fit(:,:,g-1)}, 1, nClus);
    Q0Filt = blkdiag(Qc{:});
    C_trans_tmp = zeros(n*nClus, p*nClus);
    for k = 1:length(Lab)
        C_trans_tmp(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,g-1);
    end
    
    [Xs,Ws,~] = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_fit(:,g-1),...
        x0_fit(:,g-1),Q0Filt*1e3,A_fit(:,:,g-1),b_fit(:,g-1),Q_fit(:,:,g-1));
    X_fit(:,:,g) = mvnrnd(Xs',round(Ws, 6))';
    
%     plot(X_fit(:,:,g)')
    
    % (2) update d: do things block-wise/reorder for clustering: PASS
    % here just don't do that for simplicity
    lamd = @(d) exp(C_trans_tmp*X_fit(:,:,g) + d);
    
    derd = @(d) sum(Y - lamd(d), 2) - inv(Sigd0)*(d - mud0);
    hesd = @(d) -diag(sum(lamd(d),2)) - inv(Sigd0);
    [mud,~,niSigd,~] = newton(derd,hesd,d_fit(:,g-1),1e-6,1000);
    Sigd = -inv(niSigd);
    d_fit(:,g) = mvnrnd(mud,Sigd)';
    
%     [d_fit(:,g) d]
    
    % (3) update C: must do things block-wise here... PASS
    % depends on identifiabiility of latent...
    for l = unique(Lab)
        latentId = ((l-1)*p+1):(l*p);
        C_tmp = C_fit(Lab == l,:,g-1);
        d_tmp = d_fit(Lab == l, g);
        Y_tmp = Y(Lab==l,:);
        
        derC_tmp = @(vecC) derC(vecC,Y(Lab==l,:), X_fit(latentId,:,g),...
            d_fit(Lab == l, g)) - inv(SigCj0)*(vecC - muCj0);
        
        
        hessC_tmp = @(vecC) hessC(vecC, X_fit(latentId,:,g),...
            d_fit(Lab == l, g)) - inv(SigCj0);
        [muC,~,niSigC,~] = newton(derC_tmp,hessC_tmp,...
            C_tmp(:),1e-8,1000);
        
%         [reshape(muC, [], p) C_all(Lab == l,:)]
        
        SigC = -inv(niSigC);
        C_fit(Lab == l,:,g) = reshape(mvnrnd(muC,SigC)', [], p);
    end
    
    
%     imagesc([C_fit(:,:,g)-C_all])
%     colorbar()
    
    % (4) update x0: PASS
    Qc = repmat({Q0_fit(:,:,g-1)}, 1, nClus);
    Q0expand = blkdiag(Qc{:});
    
    Sigx0 = inv(inv(Sigx00) + inv(Q0expand));
    mux0 = Sigx0*(inv(Sigx00)*mux00 + inv(Q0expand)*X_fit(:,1,g));
    x0_fit(:,g) = mvnrnd(mux0, Sigx0)';
    
%     [x0_fit(:,g) [mu1 mu2 mu3]']
    
    % (5) update Q0: assume all cluster share the same Q0: FAIL
%     XQ0 = reshape(X_fit(:,1,g) - x0_fit(:,g), [], p);
%     PsiQ0 = Psi00 + XQ0'*XQ0;
%     nuQ0 = nClus + nu00;
%     Q0_fit(:,:,g) = iwishrnd(PsiQ0,nuQ0);
    
%     [Q0_fit(:,:,g) Q0]
    
    % (6) update A: do things block-wise: PASS?
    for rec = unique(Lab)
        for send = unique(Lab)
            
            recId = ((rec-1)*p+1):(rec*p);
            sendId = ((send-1)*p+1):(send*p);
            
            z_tmp = X_fit(recId,2:T,g) -...
                repmat(b_fit(recId, g-1), 1, T-1) -...
                A(recId, setdiff(1:end, sendId))*...
                X_fit(setdiff(1:end, sendId),1:(T-1),g);
            
            z_tmp2 = z_tmp(:);
            X_tmp = kron(X_fit(sendId, 1:(T-1), g)', eye(p));
            
            
            SigA_tmp = inv(inv(SigAjl0) + X_tmp'*kron(eye(T-1), inv(Q_fit(recId,recId,g-1)))*X_tmp);
            muA_tmp = SigA_tmp*(inv(SigAjl0)*muAjl0 +...
                X_tmp'*kron(eye(T-1), inv(Q_fit(recId,recId,g-1)))*z_tmp2);
            
            A_fit(recId,sendId,g) = reshape(mvnrnd(muA_tmp, SigA_tmp)', [], p);
            
        end
    end
    
%     imagesc(A_fit(:,:,g) - A)
    
    
    % (7) update b: PASS
    r_tmp = X_fit(:,2:T,g) - A_fit(:,:,g)*X_fit(:,1:(T-1),g);
    Sigb = inv(inv(Sigb0) + (T-1)*inv(Q_fit(:,:,g-1)));
    mub = Sigb*(inv(Sigb0)*mub0 + (T-1)*inv(Q_fit(:,:,g-1))*mean(r_tmp, 2));
    b_fit(:,g) = mvnrnd(mub, Sigb)';
    
%     [b_fit(:,g) b]
    
    % (8) update Q: do things block-wise? PASS
%     for l = unique(Lab)
%         latentId = ((l-1)*p+1):(l*p);
%         mux = A_fit(latentId,:,g)*X_fit(:,1:(T-1),g) + repmat(b_fit(latentId, g), 1, T-1);
%         xq = X_fit(latentId,2:T,g) - mux;
%         
%         PsiQ = Psi0 + xq*xq';
%         nuQ = T-1 + nu0;
%         Q_fit(latentId,latentId,g) = iwishrnd(PsiQ,nuQ);
%     end
    
%     imagesc(Q_fit(:,:,g))
%     colorbar()
%     imagesc(Q - Q_fit(:,:,g))
%     colorbar()
end

for g = 1:10
    figure(g)
    plot(X_fit(:,:,g)')
end


















