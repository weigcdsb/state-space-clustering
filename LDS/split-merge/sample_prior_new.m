function theta =...
    sample_prior_new(prior, N, T, p, sampleDynamics)

% lodaing: C
theta.C = randn(N, p);

% linear dynamics: b, A & Q
theta.A = eye(p+1);
theta.b = zeros(p+1,1);
theta.Q = eye(p+1)*prior.Psi0;

% mean & latent: d, X
dX = ones(1+p,T)*Inf;
dX(:,1) = mvnrnd(prior.theta0, prior.Q0);

flag = 1;
while flag
    if sampleDynamics
        for k = 1:p+1
            theta.Q(k,k)= iwishrnd(prior.Psi0, prior.nu0);
            baSamp = mvnrnd(prior.BA0, kron(theta.Q(k,k), inv(prior.Lamb0)))';
            theta.b(k) = baSamp(1);
            theta.A(k,k) = baSamp(2);
        end
    end
    
    try
        for t= 2:T
            dX(:, t) = mvnrnd(theta.b + theta.A*dX(:, t-1), theta.Q);
        end
        theta.d = dX(1,:);
        theta.X = dX(2:end,:);
        flag = 0;
    catch
        flag = 1;
    end
end


end