function theta =...
    sample_prior_new(prior, N, T, p, sampleDynamics, maxCount)

flag = 1;
while flag
    % lodaing: C
    theta.C = randn(N, p);
    theta.muC = theta.C;
    
    % linear dynamics: b, A & Q
    theta.A = eye(p+1);
    theta.b = zeros(p+1,1);
    theta.Q = eye(p+1)*prior.Psi0;
    
    % mean & latent: d, X
    dX = ones(1+p,T)*Inf;
    dX(:,1) = mvnrnd(prior.theta0, prior.Q0);
    
    if sampleDynamics
        for k = 1:p+1
            theta.Q(k,k)= iwishrnd(prior.Psi0, prior.nu0);
            baSamp = mvnrnd(prior.BA0, kron(theta.Q(k,k), inv(prior.Lamb0)))';
            theta.b(k) = baSamp(1);
            theta.A(k,k) = baSamp(2);
        end
    end
    
    for t= 2:T
        dX(:, t) = mvnrnd(theta.b + theta.A*dX(:, t-1), theta.Q);
    end
    
    theta.d = dX(1,:);
    theta.X = dX(2:end,:);
    
    lamTmp = exp([ones(N,1) theta.C]*[theta.d ;theta.X]);
    if(sum(isinf(lamTmp(:))) ==0 && max(lamTmp(:)) <= maxCount)
       flag = 0; 
    end
    
end


end