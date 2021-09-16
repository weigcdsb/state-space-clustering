

rng(2)
tic;
for g = 2:50
    
    disp(g)
    
    % (1) update X_fit
    % adaptive smoothing
    
    C_trans_tmp = zeros(N, p*nClus);
    for k = 1:N
        C_trans_tmp(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,g-1);
    end
    d_tmp = d_fit(:,g-1);
    x0_tmp = x0_fit(:,g-1);
    A_tmp = A_fit(:,:,g-1);
    b_tmp = b_fit(:,g-1);
    Q_tmp = Q_fit(:,:,g-1);
    X_tmp = ppasmoo_poissexp_v2(Y,C_trans_tmp,d_tmp,x0_tmp,Q0,A_tmp,b_tmp,Q_tmp);
    
    logpdf = @(vecX)logpdfX(vecX, d_tmp, C_trans_tmp, x0_tmp,...
        Q0_tmp, Q_tmp, A_tmp, b_tmp, Y);
    
%     smp = hmcSampler(logpdf,muXvec, 'CheckGradient',0, 'MassVector',diag(-hess_tmp));
    smp = hmcSampler(logpdf,X_tmp(:), 'CheckGradient',0);
    muXvec_HMC = drawSamples(smp,'Burnin',0,'NumSamples',1);
    
    X_fit(:,:,g) = reshape(muXvec_HMC,[], T);
    
end
toc;

x_norm = zeros(g, 1);
for k = 1:g
    x_norm(k) = norm(X_fit(:,:,k), 'fro');
end

plot(x_norm)



