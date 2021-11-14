function [theta_b, epsilonOut, log_pdf] =...
    update_cluster2(Y_tmp,theta_a,theta_b,...
    prior, N, T, p, obsIdx, active, density, OPTDC_tmp)

% for debug
% obsIdx = obsIdx;
% Y_tmp = Y(obsIdx,:);
% theta_a = THETA{g-1}(c);
% theta_b = THETA{g}(c);
% active = true;
% density = false;
% OPTDC_tmp = OPTDC(obsIdx);

N_tmp = size(Y_tmp, 1);
log_pdf = NaN;

% (1) update X_fit
gradHess = @(vecX) gradHessX(vecX, theta_a.d(obsIdx), theta_a.C(obsIdx,:),...
    theta_a.x0, prior.Q0, theta_a.Q, theta_a.A, theta_a.b, Y_tmp);
[muXvec,~,hess_tmp,~] = newtonGH(gradHess,theta_a.X(:),1e-6,1000);

if(sum(isnan(muXvec)) ~= 0)
    disp('use adaptive smoother initial')
    X_tmp = ppasmoo_poissexp_v2(Y_tmp,theta_a.C(obsIdx,:),...
        theta_a.d(obsIdx),theta_a.x0,prior.Q0,theta_a.A,theta_a.b,theta_a.Q);
    [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-6,1000);
end

% use Cholesky decomposition to sample efficiently
if active
    RX = chol(-hess_tmp,'lower'); % sparse
    zX = randn(length(muXvec), 1) + RX'*muXvec;
    Xsamp = RX'\zX;
    theta_b.Xori = reshape(Xsamp,[], T);
    
%     theta_b.X = theta_b.Xori - mean(theta_b.Xori, 2);
%     [UX, ~, VX] = svd(theta_b.X', 'econ');
%     theta_b.X = VX*UX';

%     [QX, ~] = mgson(theta_b.X');
%     theta_b.X = QX';
end

if density
    log_pdf = 0;
%     log_pdf = log_pdf + mvnlpdf(theta_b.Xori(:), muXvec, -hess_tmp);
end


% (2) update x0_fit
% invSigx0 = sparse(inv(prior.Sigx00) + inv(prior.Q0));
% mux0 = invSigx0\(prior.Sigx00\prior.mux00 + prior.Q0\theta_b.X(:,1));
% if active
%     R = chol(invSigx0,'lower'); % sparse
%     z = randn(length(mux0), 1) + R'*mux0;
%     theta_b.x0 = R'\z;
% end
% 
% if density
%     log_pdf = log_pdf + mvnlpdf(theta_b.x0, mux0, invSigx0);
% end

% (3) update d_fit & C_fit
% the transition kernel for NUTS is symmetric --> no need for calculation
dOut_obs = zeros(N_tmp, 1);
COut_obs = zeros(N_tmp, p);
epsilonOut = ones(N_tmp, 1)*0.01;

if active
    for i = 1:N_tmp
%         X_tmpdc = [ones(1, T) ; theta_b.X]';
        X_tmpdc = [ones(1, T) ; theta_b.Xori]';
        
        lamdc = @(dc) exp(X_tmpdc*dc);
        lpdf = @(dc) sum(log(poisspdf(Y_tmp(i,:)', lamdc(dc)))) +...
            log(mvnpdf(dc, theta_a.mudc, theta_a.Sigdc));
        glpdf = @(dc) X_tmpdc'*(Y_tmp(i,:)' - lamdc(dc)) - theta_a.Sigdc\(dc - theta_a.mudc);
        fg=@(dc_r) deal(lpdf(dc_r'), glpdf(dc_r')'); % log density and gradient
        dc0 = [theta_a.d(obsIdx(i)) theta_a.C(obsIdx(i),:)]';
        
        [dc_NUTS, ~, diagn]=hmc_nuts(fg, dc0',OPTDC_tmp{i});
        epsilonOut(i) = diagn.opt.epsilonbar;
        
        dOut_obs(i) = dc_NUTS(end,1);
        COut_obs(i,:) = dc_NUTS(end,2:end);
    end
    
    gtmp = -mean(theta_b.Xori, 2);
    [UX, SX, VX] = svd(theta_b.Xori' - mean(theta_b.Xori, 2)', 'econ');
    M = VX*inv(SX)*VX';
    gtrans = M*gtmp;
    
    theta_b.X = M*theta_b.Xori + gtrans;
    dOut_obs = dOut_obs - (COut_obs/M)*gtrans;
    COut_obs = COut_obs/M;
    
else
    dOut_obs = theta_b.d(obsIdx);
    COut_obs = theta_b.C(obsIdx, :);
end

% (4) update mudc_fit & Sigdc_fit
dc_tmp = [dOut_obs COut_obs];
invTaudc = inv(prior.Taudc0) + N_tmp*inv(theta_a.Sigdc);
deltadc = invTaudc\(prior.Taudc0\prior.deltadc0 + theta_a.Sigdc\sum(dc_tmp,1)');
if active;theta_b.mudc = mvnrnd(deltadc, inv(invTaudc))';end
if density
    log_pdf = log_pdf + mvnlpdf(theta_b.mudc, deltadc, invTaudc);
end

% different covariances
dcRes = dc_tmp' - theta_b.mudc;
Psidc = prior.Psidc0 + dcRes*dcRes';
nudc = N_tmp + prior.nudc0;
if active; theta_b.Sigdc = iwishrnd(Psidc,nudc);end
if density
    log_pdf = log_pdf + iwishlpdf(theta_b.Sigdc, Psidc, nudc);
end

if active
    dc_samp = mvnrnd(theta_b.mudc, theta_b.Sigdc, N);
    theta_b.d = dc_samp(:,1);
    theta_b.d(obsIdx) = dOut_obs;
    theta_b.C = dc_samp(:,2:end);
    theta_b.C(obsIdx, :) = COut_obs;
    
    theta_b.dExpand = dc_samp(:,1);
    theta_b.CExpand = dc_samp(:,2:end);
    
end

if density
    unobsIdx = setdiff(1:N, obsIdx);
    if ~isempty(unobsIdx)
        for kk = unobsIdx
            dc_tmp = [theta_b.d(kk) theta_b.C(kk,:)]';
            log_pdf = log_pdf + mvnlpdf(dc_tmp, theta_b.mudc, inv(theta_b.Sigdc));
        end
    end
end


for k = 1:p
    Y_BA = theta_b.X(k,2:T)';
    X_BA = [ones(T-1,1) theta_b.X(k,1:(T-1))'];
    
    BAn = (X_BA'*X_BA + prior.Lamb0)\(X_BA'*Y_BA + prior.Lamb0*prior.BA0);
    PsiQ = prior.Psi0 + (Y_BA - X_BA*BAn)'*(Y_BA - X_BA*BAn) +...
        (BAn - prior.BA0)'*prior.Lamb0*(BAn - prior.BA0);
    nuQ = T-1 + prior.nu0;
    if active;theta_b.Q(k,k) = iwishrnd(PsiQ,nuQ);end
    if density
        log_pdf = log_pdf + iwishlpdf(theta_b.Q(k,k), PsiQ,nuQ);
    end
    
    % (6) update b_fit & A_fit
    Lambn = X_BA'*X_BA + prior.Lamb0;
    if active
        BAsamp = mvnrnd(BAn(:), kron(theta_b.Q(k,k), inv(Lambn)))';
        theta_b.b(k) = BAsamp(1);
        theta_b.A(k,k) = BAsamp(2);
    end
    if density
        baTmp = [theta_b.b(k) theta_b.A(k,k)]';
        log_pdf = log_pdf + mvnlpdf(baTmp, BAn(:), kron(inv(theta_b.Q(k,k)), Lambn));
    end
    
end

end