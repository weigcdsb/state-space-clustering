function [theta_b, epsilonOut, log_pdf] =...
    update_cluster_new(Y_tmp,theta_a,theta_b,...
    prior, N, T, p, obsIdx, active, density, OPTDC_tmp)

% for debug
% c=j;
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

gradHess = @(vecdX) gradHessX_bt(vecdX, zeros(N_tmp,1),...
    [ones(N_tmp,1) theta_a.C(obsIdx,:)],...
    prior.theta0, prior.Q0, theta_a.Q, theta_a.A, theta_a.b, Y_tmp);

dX = [theta_a.d;theta_a.X];
[mudXvec,~,hess_tmp,~, eoi] = newtonGH(gradHess,dX(:),1e-6,1000);

if((sum(isnan(mudXvec)) ~= 0) || eoi == 1000)
    disp('use adaptive smoother initial')
    try
        dX_tmp = ppasmoo_poissexp_bt(Y_tmp,[ones(N_tmp,1) theta_a.C(obsIdx,:)],...
            zeros(N_tmp,1),prior.theta0,prior.Q0,theta_a.A,theta_a.b,theta_a.Q);
    catch
        dX_tmp = dX*0;
    end
    
    [mudXvec,~,hess_tmp,~] = newtonGH(gradHess,dX_tmp(:),1e-6,1000);
end

% use Cholesky decomposition to sample efficiently
if active
    try
        RdX = chol(-hess_tmp,'lower'); % sparse
        zdX = randn(length(mudXvec), 1) + RdX'*mudXvec;
        dXsamp = RdX'\zdX;
    catch
        dXsamp = mudXvec;
    end
    dXsamp_trans = reshape(dXsamp,[], T);
    
    theta_b.d = dXsamp_trans(1,:);
    theta_b.X = dXsamp_trans(2:end,:);
end

if density
    log_pdf = 0;
%     log_pdf = log_pdf + mvnlpdf(theta_b.Xori(:), muXvec, -hess_tmp);
end


% (2) update loading: C
% the transition kernel for NUTS is symmetric --> no need for calculation
epsilonOut = ones(N_tmp, 1)*0.01;

if active
    for i = 1:N_tmp
        X_tmpC = theta_b.X';
        
        lamC = @(c) exp(theta_b.d' + X_tmpC*c);
        
        % use NUTS
        lpdf = @(c) sum(log(poisspdf(Y_tmp(i,:)', lamC(c)))) +...
            log(mvnpdf(c, prior.muC0, prior.SigC0));
        glpdf = @(c) X_tmpC'*(Y_tmp(i,:)' - lamC(c)) - prior.SigC0\(c - prior.muC0);
        
        fg=@(dc_r) deal(lpdf(dc_r'), glpdf(dc_r')'); % log density and gradient
        c0 = theta_a.C(obsIdx(i),:)';
        
        [c_NUTS, ~, diagn]=hmc_nuts(fg, c0',OPTDC_tmp{i});
        epsilonOut(i) = diagn.opt.epsilonbar;
        theta_b.C(obsIdx(i),:) = c_NUTS(end,:);
    end
end

dX = [theta_b.d;theta_b.X];

% (3) update A
for k = 1:(p+1)
    Y_A = dX(k,2:T)' - theta_a.b(k, 1:(T-1))';
    X_A = dX(k,1:(T-1))';
    
    sig2An = inv(inv(prior.sig2A0) + X_A'*X_A/theta_a.Q(k,k));
    muAn = sig2An*(inv(prior.sig2A0)*prior.A0 + X_A'*Y_A/theta_a.Q(k,k));
    theta_b.A(k,k) = mvnrnd(muAn, sig2An);
    
    % (4) update b
    for t = 1:(T-1)
        xbt =  dX(k,t+1) - theta_b.A(k,k)*dX(k,t);
        
        sig2bn = inv(1/prior.sig2b0 + 1/theta_a.Q(k,k));
        mubn = sig2bn*(prior.b0/prior.sig2b0 + xbt/theta_a.Q(k,k));
        theta_b.b(k,t) = normrnd(mubn,sig2bn);
    end
    
    % (5) update Q
    alphq = (prior.nu0 + T-1)/2;
    betaq = (prior.nu0*prior.sig20 +...
         sum((Y_A - X_A*theta_b.A(k,k) - theta_b.b(k,:)').^2))/2;
    theta_b.Q(k,k) = 1/gamrnd(alphq, 1/betaq);
end




end