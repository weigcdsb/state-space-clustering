addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
% addpath(genpath('D:\github\state-space-clustering'));

%% simulation
rng(1)
n = 10;
nClus = 3;
N = n*nClus;
p = 2;
T = 1000;

Lab = repelem(1:nClus, n);
pLab = repelem(1:nClus, p);

d = randn(n*nClus,1)*0;
C_trans = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = sum(Lab(1:k)==Lab(k))/sum(Lab==Lab(k))+1;
end

X = zeros(p*nClus, T);
x0 = zeros(p*nClus, 1);
Q0 = eye(nClus*p)*1e-2;
X(:,1) = mvnrnd(x0, Q0)';

b1 = ones(p,1)*0;
b2 = ones(p,1)*0;
b3 = ones(p,1)*0;
b = [b1;b2;b3];

Q1 = 1e-3*eye(p);
Q2 = 1e-3*eye(p);
Q3 = 1e-3*eye(p);
Q = blkdiag(Q1, Q2, Q3);

%
A = eye(size(Q,1));
while any(imag(eig(A))==0)
    A= randn(size(Q));
    A = A-diag(diag(A));
    A(squareform(pdist(pLab'))==0)=0;
    A = A./sqrt(sum((A-diag(diag(A))).^2,2))*0.1;
    A = A+eye(size(Q,1))*0.92;
end

% let's generate lambda
logLam = zeros(n*nClus, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    X(:, t) = mvnrnd(A*X(:, t-1) + b, Q)';
    logLam(:, t) = d + C_trans*X(:,t);
end

Y = poissrnd(exp(logLam));

figure(1)
imagesc(A)
colorbar()
xlabel('sending')
ylabel('receiving')

figure(2)
imagesc(exp(logLam))
colorbar()

figure(3)
clusterPlot(Y, Lab)

figure(4)
subplot(1,3,1)
plot(X(1:p,:)')
subplot(1,3,2)
plot(X(p+1:2*p,:)')
subplot(1,3,3)
plot(X(2*p+1:3*p,:)')

%%
% nClus = 1;
% Lab = ones(1,N);
rng(3)
ng = 10000;

% pre-allocation
X_fit = zeros(nClus*p, T, ng);
d_fit = zeros(N, nClus, ng);
C_fit = zeros(N, nClus*p, ng);
mudc_fit = zeros(p+1, nClus, ng);
Sigdc_fit = zeros(p+1, p+1, nClus, ng);
x0_fit = zeros(nClus*p, ng);
A_fit = zeros(nClus*p, nClus*p, ng);
b_fit = zeros(nClus*p, ng);
Q_fit = zeros(nClus*p, nClus*p, ng);


% priors
% place-holder...
Q0 = eye(nClus*p)*1e-2;

mux00 = zeros(nClus*p, 1);
Sigx00 = eye(nClus*p);

deltadc0 = zeros(p+1,1);
Taudc0 = eye(p+1);

Psidc0 = eye(p+1)*1e-4;
nudc0 = p+1+2;

BA0_all = [zeros(nClus*p,1) eye(nClus*p)]';
Lamb0 = eye(nClus*p + 1)*.25;
Psi0 = eye(p)*1e-4;
nu0 = p+2;

% initials
% initial for d_fit: 0
C_raw = reshape(normrnd(0,1e-2,N*p,1), [], p);
d_tmp = zeros(N,1);
for k = unique(Lab)
    ladid_tmp = id2id(k, p);
    C_fit(Lab == k, ladid_tmp, 1) = C_raw(Lab == k, :);
    d_tmp(Lab == k) = d_fit(Lab ==k, k);
end

% mudc_fit(:,:,1) = zeros(p+1, 1);
Sigdc_fit(:,:,1:nClus,1) = repmat(eye(p+1)*1e-2,1,1,nClus);

A_fit(:,:,1) = eye(nClus*p);
% initial for b_fit: 0
Q_fit(:,:,1) = eye(nClus*p)*1e-4;


x0_fit(:,1) = lsqr(C_fit(:,:,1),(log(mean(Y(:,1:10),2))-d_tmp));
[X_tmp,~,~] = ppasmoo_poissexp_v2(Y,C_fit(:,:,1),d_tmp,...
    x0_fit(:,1),Q0,A_fit(:,:,1),b_fit(:,1),Q_fit(:,:,1));
gradHess = @(vecX) gradHessX(vecX, d_tmp, C_fit(:,:,1), x0_fit(:,1), Q0,...
    Q_fit(:,:,1), A_fit(:,:,1), b_fit(:,1), Y);
[muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-10,1000);
X_fit(:,:,1) = reshape(muXvec, [], T);


%% MCMC
dc_all = [sum(d_fit(:,:,1), 2) sum(C_fit(:,1:p:end), 2) sum(C_fit(:,2:p:end), 2)]';
acc = zeros(N,1);
x_all = reshape(X_fit(:,:,1),[],1);

% optdc.M=0;
% optdc.Madapt=50;
% optdc.delta = 0.8;
% epsilon = zeros(N,1);
% 
% for i = 1:N
%     l = Lab(i);
%     latentId = id2id(l,p);
%     X_tmp = [ones(1, T) ;X_fit(latentId,:,1)]';
%     
%     lamdc = @(dc) exp(X_tmp*dc);
%     derdc = @(dc) X_tmp'*(Y(i,:)' - lamdc(dc)) - Sigdc_fit(:,:,l,1)\(dc - mudc_fit(:,l,1));
%     lpdf_dc = @(dc) sum(log(poisspdf(Y(i,:)', lamdc(dc)))) +...
%         log(mvnpdf(dc, mudc_fit(:,l,1), Sigdc_fit(:,:,l,1)));
%     
%     fg_dc=@(dc_r) deal(lpdf_dc(dc_r'), derdc(dc_r')'); % log density and gradient
%     
%     [mudc_NUTS, ~, diagn]=hmc_nuts(fg_dc, [d_fit(i,l, 1) C_fit(i,latentId,1)],optdc);
%     epsilon(i) = diagn.opt.epsilonbar;
%     
%     d_fit(i,l,1) = mudc_NUTS(end,1);
%     C_fit(i,latentId,1) = mudc_NUTS(end,2:end);
%     
% end
clear optdc
optdc.M=1;
optdc.Madapt=0;
epsilon = zeros(N,1);
burnIn = 1000;
flg = 0;

for g = 2:ng
    
    % disp(g)
    
    % (1) update X_fit
    % adaptive smoothing
    
    d_raw = d_fit(:,:,g-1);
    I = (1 : size(d_raw, 1)) .';
    k = sub2ind(size(d_raw), I, Lab');
    d_tmp = d_raw(k);
    C_tmp = C_fit(:,:,g-1);
    
    x0_tmp = x0_fit(:,g-1);
    A_tmp = A_fit(:,:,g-1);
    b_tmp = b_fit(:,g-1);
    Q_tmp = Q_fit(:,:,g-1);
    X_tmp = X_fit(:,:,g-1);
    
    gradHess = @(vecX) gradHessX(vecX, d_tmp, C_tmp, x0_tmp, Q0, Q_tmp, A_tmp, b_tmp, Y);
    [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-8,1000);
    if(sum(isnan(muXvec)) ~= 0)
        disp('use adaptive smoother initial')
        X_tmp = ppasmoo_poissexp_v2(Y,C_tmp,d_tmp,x0_tmp,Q0,A_tmp,b_tmp,Q_tmp);
        [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-8,1000);
    end
    
    R = chol(-hess_tmp,'lower'); % sparse
    z = randn(length(muXvec), 1) + R'*muXvec;
    x_all = R'\z;
    X_fit(:,:,g) = reshape(x_all,[], T);
    
%     disp("iter " + g + ": " + norm(X_fit(:,:,g) - X_fit(:,:,g-1), 'fro')/...
%                 norm(X_fit(:,:,g-1), 'fro'));
    
    % (2) update x0_fit
    Sigx0 = inv(inv(Sigx00) + inv(Q0));
    mux0 = Sigx0*(Sigx00\mux00 + Q0\X_fit(:,1,g));
    x0_fit(:,g) = mvnrnd(mux0, Sigx0)';
    % disp(x0_fit(:,g))
    
    % (3) update d_fit & C_fit
    % NUTS
    if(g < 50)
        tuneState = 1; % change epsilon
        disp("iter " + g + ": " + norm(X_fit(:,:,g) - X_fit(:,:,g-1), 'fro')/...
            norm(X_fit(:,:,g-1), 'fro') + ", changing");
    elseif(flg == 1)
        tuneState = 3; % fix epsilon
        disp("iter " + g + ": " + norm(X_fit(:,:,g) - X_fit(:,:,g-1), 'fro')/...
                    norm(X_fit(:,:,g-1), 'fro') + ", tuned");
    else
        if(g > burnIn ||...
                norm(X_fit(:,:,g) - X_fit(:,:,g-1), 'fro')/...
                norm(X_fit(:,:,g-1), 'fro') < 1e-1)
            tuneState = 2; % tune epsilon
            disp("iter " + g + ": " + norm(X_fit(:,:,g) - X_fit(:,:,g-1), 'fro')/...
                    norm(X_fit(:,:,g-1), 'fro') + ", tuning");
        else
            tuneState = 1; % change epsilon
            disp("iter " + g + ": " + norm(X_fit(:,:,g) - X_fit(:,:,g-1), 'fro')/...
            norm(X_fit(:,:,g-1), 'fro') + ", changing");
        end
    end
    
    
    
    for i = 1:N
        l = Lab(i);
        latentId = id2id(l,p);
        X_tmp = [ones(1, T) ;X_fit(latentId,:,g)]';
        
        lamdc = @(dc) exp(X_tmp*dc);
        derdc = @(dc) X_tmp'*(Y(i,:)' - lamdc(dc)) - Sigdc_fit(:,:,l,g-1)\(dc - mudc_fit(:,l,g-1));
        lpdf_dc = @(dc) sum(log(poisspdf(Y(i,:)', lamdc(dc)))) +...
            log(mvnpdf(dc, mudc_fit(:,l,g-1), Sigdc_fit(:,:,l,g-1)));
        
        fg_dc=@(dc_r) deal(lpdf_dc(dc_r'), derdc(dc_r')'); % log density and gradient
        
        switch tuneState
            case 1
                [mudc_NUTS, ~, ~]=hmc_nuts(fg_dc, [d_fit(i,l, g-1) C_fit(i,latentId,g-1)],optdc);
            case 2
                optdc.Madapt=50;
                [mudc_NUTS, ~, diagn]=hmc_nuts(fg_dc, [d_fit(i,l, g-1) C_fit(i,latentId,g-1)],optdc);
                epsilon(i) = diagn.opt.epsilonbar;
                optdc.Madapt=0;
            case 3
                optdc.epsilon = epsilon(i);
                [mudc_NUTS, ~, ~]=hmc_nuts(fg_dc, [d_fit(i,l, g-1) C_fit(i,latentId,g-1)],optdc);
        end
        
        d_fit(i,l,g) = mudc_NUTS(2,1);
        C_fit(i,latentId,g) = mudc_NUTS(2,2:end);
        
    end
    
    if(tuneState == 2);flg = 1;end    
    
    % (4) update mudc_fit & Sigdc_fit
    %    dcRes_all = [];
    for l = unique(Lab)
        dc_tmp = [d_fit(Lab == l, l, g) C_fit(Lab == l, id2id(l,p), g)];
        invTaudc = inv(Taudc0) + sum(Lab == l)*inv(Sigdc_fit(:,:,l,g-1));
        deltadc = invTaudc\(Taudc0\deltadc0 + Sigdc_fit(:,:,l,g-1)\sum(dc_tmp,1)');
        mudc_fit(:,l,g) = mvnrnd(deltadc, inv(invTaudc));
        
        % assume different covariances
        dcRes = dc_tmp' - mudc_fit(:,l,g);
        Psidc = Psidc0 + dcRes*dcRes';
        nudc = sum(Lab == l) + nudc0;
        Sigdc_fit(:,:,l,g) = iwishrnd(Psidc,nudc);
        
        % assume single covariance
        %         dcRes_all = [dcRes_all dc_tmp' - mudc_fit(:,l,g)];
    end
    
    % assume single covariance
    %     Psidc = Psidc0 + dcRes_all*dcRes_all';
    %     nudc = N + nudc0;
    %     Sigdc_fit(:,:,:,g) = repmat(iwishrnd(Psidc,nudc),1,1,nClus);
    
    %     mudc_fit(:,:,g)
    %     Sigdc_fit(:,:,:,g)
    
    for l = unique(Lab)
        latentId = id2id(l,p);
        
        % (4)update Q
        Y_BA = X_fit(latentId,2:T,g)';
        X_BA = [ones(T-1,1) X_fit(:,1:(T-1),g)'];
        
        BA0 = BA0_all(:,latentId);
        BAn = (X_BA'*X_BA + Lamb0)\(X_BA'*Y_BA + Lamb0*BA0);
        PsiQ = Psi0 + (Y_BA - X_BA*BAn)'*(Y_BA - X_BA*BAn) +...
            (BAn - BA0)'*Lamb0*(BAn - BA0);
        nuQ = T-1 + nu0;
        Q_fit(latentId,latentId,g) = iwishrnd(PsiQ,nuQ);
        
        % (5) update b_fit & A_fit
        Lambn = X_BA'*X_BA + Lamb0;
        BAvec = mvnrnd(BAn(:), kron(Q_fit(latentId,latentId,g), inv(Lambn)))';
        BAsamp = reshape(BAvec,[], p)';
        b_fit(latentId,g) = BAsamp(:,1);
        A_fit(latentId,:,g) = BAsamp(:,2:end);
    end
    
    
    figure(1)
    subplot(3,2,1)
    plot(X(1:p,:)')
    title('true')
    subplot(3,2,2)
    plot(X_fit(1:p,:,g)')
    title('fit')
    subplot(3,2,3)
    plot(X(p+1:2*p,:)')
    subplot(3,2,4)
    plot(X_fit(p+1:2*p,:,g)')
    subplot(3,2,5)
    plot(X(2*p+1:3*p,:)')
    subplot(3,2,6)
    plot(X_fit(2*p+1:3*p,:,g)')
    
end

save('C:\Users\gaw19004\Desktop\LDS_backup\new2\lds_NUTS.mat')

%%

% accx/(ng-x_MHStart)
% acc/(ng-dc_MHStart)

d_norm = zeros(g, 1);
C_norm_fro = zeros(g, 1);
b_norm = zeros(g, 1);
A_norm_fro = zeros(g, 1);
X_norm_fro = zeros(g, 1);

for k = 1:g
    d_norm(k) = norm(d_fit(:,:,k), 'fro');
    C_norm_fro(k) = norm(C_fit(:,:,k), 'fro');
    b_norm(k) = norm(b_fit(:,k));
    A_norm_fro(k) = norm(A_fit(:,:,k), 'fro');
    X_norm_fro(k) = norm(X_fit(:,:,k), 'fro');
end

x_MHStart = 1;
figure
plot(X_norm_fro(x_MHStart:end))
% xline(x_MHStart, 'r')
title("Frobenius norm of X, dim: " + nClus*p + "\times" + T)


figure
subplot(2,2,1)
plot(d_norm(x_MHStart:end))
% xline(x_MHStart, 'r')
title('norm of d')
subplot(2,2,2)
plot(C_norm_fro(x_MHStart:end))
% xline(x_MHStart, 'r')
title('Frobenius norm of C')
subplot(2,2,3)
plot(b_norm(x_MHStart:end))
% xline(x_MHStart, 'r')
title('norm of b')
subplot(2,2,4)
plot(A_norm_fro(x_MHStart:end))
% xline(x_MHStart, 'r')
title('Frobenius norm of A')


%%
idx = round(ng/2):ng;

subplot(1,2,1)
imagesc(exp(C_trans*X + d))
cLim = caxis;
title('true')
colorbar()
subplot(1,2,2)
imagesc(exp(mean(C_fit(:,:,idx), 3)*mean(X_fit(:,:,idx), 3) + sum(mean(d_fit(:,:,idx), 3),2)))
set(gca,'CLim',cLim)
title('fit')
colorbar()

% plot(mean(X_fit(:,:,idx), 3)')

subplot(3,2,1)
plot(X(1:p,:)')
title('true')
subplot(3,2,2)
plot(mean(X_fit(1:p,:,idx), 3)')
title('fit')
subplot(3,2,3)
plot(X(p+1:2*p,:)')
subplot(3,2,4)
plot(mean(X_fit(p+1:2*p,:,idx), 3)')
subplot(3,2,5)
plot(X(2*p+1:3*p,:)')
subplot(3,2,6)
plot(mean(X_fit(2*p+1:3*p,:,idx), 3)')

subplot(1,2,1)
imagesc(A)
colorbar()
cLim = caxis;
title('true')
subplot(1,2,2)
imagesc(mean(A_fit(:,:,idx), 3))
colorbar()
set(gca,'CLim',cLim)
title('fit')

mean(d_fit(:,:,idx), 3)
mean(C_fit(:,:,idx), 3)
mean(Q_fit(:,:,idx), 3)
mean(mudc_fit(:,:,idx), 3)
mean(Sigdc_fit(:,:,:,idx), 4)

figure
subplot(1,3,1)
plot(d,sum(mean(d_fit(:,:,idx), 3),2),'rx');
ylabel('estimate')
title('d')
C_fit_mean = mean(C_fit(:,:,idx), 3);
subplot(1,3,2)
plot(sum(C_trans(:,1:p:end), 2),sum(C_fit_mean(:,1:p:end), 2),'rx');
xlabel('true')
title('1st column of C')
subplot(1,3,3)
plot(sum(C_trans(:,2:p:end), 2),sum(C_fit_mean(:,2:p:end), 2),'rx');
title('2nd column of C')
