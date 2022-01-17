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

        % ues MH
%         derc = @(c) X_tmpC'*(Y_tmp(i,:)' - lamC(c)) - prior.SigC0\(c - prior.muC0);
%         hessc = @(c) -X_tmpC'*diag(lamC(c))*X_tmpC - inv(prior.SigC0);
%         c0 = theta_a.C(obsIdx(i),:)';
%         
%         [muc,~,niSigc,~] = newton(derc,hessc,c0,1e-8,1000);
%         if(sum(isnan(muc)) ~= 0)
%             disp('use 0')
%             [muc,~,niSigc,~] = newton(derc,hessc,zeros(size(c0)),1e-8,1000);
%         end
%         
%         
%         R = chol(-niSigc,'lower'); % sparse
%         z = randn(length(c0), 1) + R'*c0;
%         cStar = R'\z;
%         
%         % lhr
%         lhr = sum(log(poisspdf(Y_tmp(i,:)', lamC(cStar)))) -...
%             sum(log(poisspdf(Y_tmp(i,:)', lamC(c0)))) +...
%             log(mvnpdf(cStar, prior.muC0, prior.SigC0)) -...
%             log(mvnpdf(c0, prior.muC0, prior.SigC0));
%         
%         if(log(rand(1)) < lhr)
%             theta_b.C(obsIdx(i),:) = cStar;
%         else
%             theta_b.C(obsIdx(i),:) = c0;
%         end
%         
%         
%         
%         Rc = chol(-niSigc,'lower'); % sparse
%         zc = randn(length(muc), 1) + Rc'*muc;
%         theta_b.C(obsIdx(i),:) = Rc'\zc;
    end
end

dX = [theta_b.d;theta_b.X];
for k = 1:(p+1)
    
    Y_A = dX(k,2:T)' - theta_a.b(k, 1:(T-1))';
    X_A = dX(k,1:(T-1))';
    
    An = (X_A'*X_A + prior.Lamb0)\(X_A'*Y_A + prior.Lamb0*prior.A0);
    PsiQ = prior.Psi0 + (Y_A - X_A*An)'*(Y_A - X_A*An) +...
        (An - prior.A0)'*prior.Lamb0*(An - prior.A0);
    nuQ = T-1 + prior.nu0;
    if active;theta_b.Q(k,k) = iwishrnd(PsiQ,nuQ);end
    if density
        log_pdf = log_pdf + iwishlpdf(theta_b.Q(k,k), PsiQ,nuQ);
    end
    
    % (3) update A_fit
    Lambn = X_A'*X_A + prior.Lamb0;
    if active
        theta_b.A(k,k) = mvnrnd(An(:), kron(theta_b.Q(k,k), inv(Lambn)))';
    end
    if density
        baTmp = [theta_b.b(k) theta_b.A(k,k)]';
        log_pdf = log_pdf + mvnlpdf(baTmp, An(:), kron(inv(theta_b.Q(k,k)), Lambn));
    end
    
end

% (3) update b_fit
for t = 1:(T-1)
    for k = 1:(p+1)
       xbt =  dX(k,t+1) - theta_b.A(k,k)*dX(k,t);
       sig2bt = theta_b.Q(k,k);
       
       Sigbt = inv(1/prior.sig2b0 + 1/sig2bt);
       mubt = Sigbt*(prior.b0/prior.sig2b0 + xbt/sig2bt);
       theta_b.b(k,t) = normrnd(mubt,Sigbt);
    end
end





end