function [theta_b, log_pdf] =...
    update_cluster_cond(Y,theta_a,theta_b,...
    prior, N, T, p, obsIdx, active, density)

% for debug
% obsIdx = obsIdx;
% theta_a = THETA{g-1}(c);
% theta_b = THETA{g}(c);
% active = true;
% density = false;
unObsIdx = setdiff(1:N, obsIdx);

Y_obs = Y(obsIdx,:);
Y_unObs = Y(unObsIdx,:);

N_obs = size(Y_obs, 1);
N_unObs = size(Y_unObs, 1);

log_pdf = NaN;

% (1) update X_fit
gradHess = @(vecX) gradHessX(vecX, theta_a.d(obsIdx), theta_a.C(obsIdx,:),...
    theta_a.x0, prior.Q0, theta_a.Q, theta_a.A, theta_a.b, Y_obs);
[muXvec,~,hess_tmp,~] = newtonGH(gradHess,theta_a.X(:),1e-6,1000);

if(sum(isnan(muXvec)) ~= 0)
    disp('use adaptive smoother initial')
    X_tmp = ppasmoo_poissexp_v2(Y_obs,theta_a.C(obsIdx,:),...
        theta_a.d(obsIdx),theta_a.x0,prior.Q0,theta_a.A,theta_a.b,theta_a.Q);
    [muXvec,~,hess_tmp,~] = newtonGH(gradHess,X_tmp(:),1e-6,1000);
end

% use Cholesky decomposition to sample efficiently
if active
    RX = chol(-hess_tmp,'lower'); % sparse
    zX = randn(length(muXvec), 1) + RX'*muXvec;
    Xsamp = RX'\zX;
    theta_b.Xori = reshape(Xsamp,[], T);
    
    theta_b.X = theta_b.Xori - mean(theta_b.Xori, 2);
    [UX, ~, VX] = svd(theta_b.X', 'econ');
    theta_b.X = VX*UX';

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
optdc.M=1;
optdc.Madapt=0;

if active
    for i = 1:N_obs
        X_tmpdc = [ones(1, T) ; theta_b.X]';
%         X_tmpdc = [ones(1, T) ; theta_b.Xori]';
        
        lamdc = @(dc) exp(X_tmpdc*dc);
        lpdf = @(dc) sum(log(poisspdf(Y_obs(i,:)', lamdc(dc)))) +...
            log(mvnpdf(dc, theta_a.mudc, theta_a.Sigdc));
        glpdf = @(dc) X_tmpdc'*(Y_obs(i,:)' - lamdc(dc)) -...
            theta_a.Sigdc\(dc - theta_a.mudc);
        fg=@(dc_r) deal(lpdf(dc_r'), glpdf(dc_r')'); % log density and gradient
        dc0 = [theta_a.d(obsIdx(i)) theta_a.C(obsIdx(i),:)]';
        
        [dc_NUTS, ~, ~]=hmc_nuts(fg, dc0',optdc);
        
        theta_b.d(obsIdx(i)) = dc_NUTS(end,1);
        theta_b.C(obsIdx(i),:) = dc_NUTS(end,2:end);
        
        
%         derdc = @(dc) X_tmpdc'*(Y_obs(i,:)' - lamdc(dc)) - theta_a.Sigdc\(dc - theta_a.mudc);
%         hessdc = @(dc) -X_tmpdc'*diag(lamdc(dc))*X_tmpdc - inv(theta_a.Sigdc);
%         dc0 = [theta_a.d(obsIdx(i)) theta_a.C(obsIdx(i),:)]';
%         
%         [mudc,~,niSigdc,~] = newton(derdc,hessdc,dc0,1e-8,1000);
%         if(sum(isnan(mudc)) ~= 0)
%             disp('use 0')
%             [mudc,~,niSigdc,~] = newton(derdc,hessdc,zeros(size(dc0)),1e-8,1000);
%         end
%         
%         % MH
%         R = chol(-niSigdc,'lower'); % sparse
%         z = randn(length(dc0), 1) + R'*dc0;
%         dcStar = R'\z;
%         
%         % lhr
%         lhr = sum(log(poisspdf(Y_obs(i,:)', lamdc(dcStar)))) -...
%             sum(log(poisspdf(Y_obs(i,:)', lamdc(dc0)))) +...
%             log(mvnpdf(dcStar, theta_a.mudc, theta_a.Sigdc)) -...
%             log(mvnpdf(dc0, theta_a.mudc, theta_a.Sigdc));
%         
%         if(log(rand(1)) < lhr)
%             theta_b.d(obsIdx(i)) = dcStar(1);
%             theta_b.C(obsIdx(i),:) = dcStar(2:end)';
%         else
%             theta_b.d(obsIdx(i)) = dc0(1);
%             theta_b.C(obsIdx(i),:) = dc0(2:end)';
%         end
    end 
end

% (4) update mudc_fit & Sigdc_fit

dc_tmp = [theta_b.d(obsIdx) theta_b.C(obsIdx,:)];
invTaudc = inv(prior.Taudc0) + N_obs*inv(theta_a.Sigdc);
deltadc = invTaudc\(prior.Taudc0\prior.deltadc0 + theta_a.Sigdc\sum(dc_tmp,1)');
invTaudc = (invTaudc + invTaudc')/2;

if active;theta_b.mudc = mvnrnd(deltadc, inv(invTaudc))';end
if density
    log_pdf = log_pdf + mvnlpdf(theta_b.mudc, deltadc, invTaudc);
end

% different covariances
dcRes = dc_tmp' - theta_b.mudc;
Psidc = prior.Psidc0 + dcRes*dcRes';
nudc = N_obs + prior.nudc0;
if active; theta_b.Sigdc = iwishrnd(Psidc,nudc);end
if density
    log_pdf = log_pdf + iwishlpdf(theta_b.Sigdc, Psidc, nudc);
end

if active
    
    for i = 1:N_unObs
        X_tmpdc = [ones(1, T) ; theta_b.X]';
%         X_tmpdc = [ones(1, T) ; theta_b.Xori]';
        
        lamdc = @(dc) exp(X_tmpdc*dc);
        
        lpdf = @(dc) sum(log(poisspdf(Y_unObs(i,:)', lamdc(dc)))) +...
            log(mvnpdf(dc, theta_b.mudc, theta_b.Sigdc));
        glpdf = @(dc) X_tmpdc'*(Y_unObs(i,:)' - lamdc(dc)) -...
            theta_b.Sigdc\(dc - theta_b.mudc);
        fg=@(dc_r) deal(lpdf(dc_r'), glpdf(dc_r')'); % log density and gradient
        dc0 = [theta_a.d(unObsIdx(i)) theta_a.C(unObsIdx(i),:)]';
        
        [dc_NUTS, ~, ~]=hmc_nuts(fg, dc0',optdc);
        
        theta_b.d(unObsIdx(i)) = dc_NUTS(end,1);
        theta_b.C(unObsIdx(i),:) = dc_NUTS(end,2:end);
        
%         derdc = @(dc) X_tmpdc'*(Y_unObs(i,:)' - lamdc(dc)) -...
%             theta_b.Sigdc\(dc - theta_b.mudc);
%         hessdc = @(dc) -X_tmpdc'*diag(lamdc(dc))*X_tmpdc - inv(theta_b.Sigdc);
%         dc0 = [theta_a.d(unObsIdx(i)) theta_a.C(unObsIdx(i),:)]';
%         [mudc,~,niSigdc,~] = newton(derdc,hessdc,dc0,1e-8,1000);
%         if(sum(isnan(mudc)) ~= 0)
%             disp('use 0')
%             [mudc,~,niSigdc,~] = newton(derdc,hessdc,zeros(size(dc0)),1e-8,1000);
%         end
%         
%         % MH
%         R = chol(-niSigdc,'lower'); % sparse
%         z = randn(length(dc0), 1) + R'*dc0;
%         dcStar = R'\z;
%         
%         % lhr
%         lhr = sum(log(poisspdf(Y_unObs(i,:)', lamdc(dcStar)))) -...
%             sum(log(poisspdf(Y_unObs(i,:)', lamdc(dc0)))) +...
%             log(mvnpdf(dcStar, theta_b.mudc, theta_b.Sigdc)) -...
%             log(mvnpdf(dc0, theta_b.mudc, theta_b.Sigdc));
%         
%         if(log(rand(1)) < lhr)
%             theta_b.d(unObsIdx(i)) = dcStar(1);
%             theta_b.C(unObsIdx(i),:) = dcStar(2:end)';
%         else
%             theta_b.d(unObsIdx(i)) = dc0(1);
%             theta_b.C(unObsIdx(i),:) = dc0(2:end)';
%         end
    end
    
    
%     gtmp = -mean(theta_b.Xori, 2);
%     [UX, SX, VX] = svd(theta_b.Xori' - mean(theta_b.Xori, 2)', 'econ');
%     M = VX*inv(SX)*VX';
%     gtrans = M*gtmp;
%     
%     theta_b.X = M*theta_b.Xori + gtrans;
%     theta_b.d = theta_b.d - (theta_b.C/M)*gtrans;
%     theta_b.C = theta_b.C/M;
%     
%     Adc = [1 (M\gtrans)'; zeros(p,1) inv(M)'];
%     theta_b.mudc = Adc*theta_b.mudc;
%     theta_b.Sigdc = Adc*theta_b.Sigdc*Adc';
    
end

if density
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