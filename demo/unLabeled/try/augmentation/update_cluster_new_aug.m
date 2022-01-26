function theta_b =...
    update_cluster_new_aug(Y_tmp,theta_a,theta_b,prior, T, p, obsIdx)

% for debug
% c=j; 
% c = actList(j);
% Y_tmp = Y(obsIdx,:);
% theta_a = THETA{g-1}(c);
% theta_b = THETA{g}(c);

N_tmp = size(Y_tmp, 1);

% (1) update X...
gradHess = @(vecdX) gradHessX(vecdX, zeros(N_tmp,1),...
    [ones(N_tmp,1) theta_a.C(obsIdx,:)],...
    prior.theta0, prior.Q0, theta_a.Q, theta_a.A, theta_a.b, Y_tmp);

dX = [theta_a.d;theta_a.X];
warning('off');
[mudXvec,~,hess_tmp,~, eoi] = newtonGH(gradHess,dX(:),1e-4,1000);
warning('on');

if((sum(isnan(mudXvec)) ~= 0) || eoi == 1000)
    try
        disp('use adaptive smoother initial')
        dX_tmp = ppasmoo_poissexp_na(Y_tmp,[ones(N_tmp,1) theta_a.C(obsIdx,:)],...
            zeros(N_tmp,1),prior.theta0,prior.Q0,theta_a.A,theta_a.b,theta_a.Q);
    catch
        disp('use 0')
        dX_tmp = dX*0;
    end
    warning('off');
    [mudXvec,~,hess_tmp,~] = newtonGH(gradHess,dX_tmp(:),1e-4,1000);
    warning('on');
end

% use Cholesky decomposition to sample efficiently
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

% (2) update linear dynamics...
dX = [theta_b.d;theta_b.X];
for k = 1:(p+1)
    try
        Y_BA = dX(k,2:T)';
        X_BA = [ones(T-1,1) dX(k,1:(T-1))'];
        
        % update Q
        BAn = (X_BA'*X_BA + prior.Lamb0)\(X_BA'*Y_BA + prior.Lamb0*prior.BA0);
        PsiQ = prior.Psi0 + (Y_BA - X_BA*BAn)'*(Y_BA - X_BA*BAn) +...
            (BAn - prior.BA0)'*prior.Lamb0*(BAn - prior.BA0);
        nuQ = T-1 + prior.nu0;
        theta_b.Q(k,k) = iwishrnd(PsiQ,nuQ);
        
        % update b & A
        Lambn = X_BA'*X_BA + prior.Lamb0;
        BAsamp = mvnrnd(BAn(:), kron(theta_b.Q(k,k), inv(Lambn)))';
        theta_b.b(k) = BAsamp(1);
        theta_b.A(k,k) = BAsamp(2);
    catch
        theta_b.Q(k,k) = theta_a.Q(k,k);
        theta_b.b(k) =   theta_a.b(k);
        theta_b.A(k,k) = theta_a.A(k,k);
    end
    
end

end