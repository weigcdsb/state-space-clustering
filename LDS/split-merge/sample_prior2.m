function theta =...
    sample_prior2(prior, N, T, p, sampleDynamics)

% baseline & lodaing: d & C
% with hyper-parameter: mudc, Sigdc
mudcOuter = mvnrnd(prior.deltadc0, prior.Taudc0);
SigdcOuter = iwishrnd(prior.Psidc0,prior.nudc0);
theta.mudc = mudcOuter';
theta.Sigdc = SigdcOuter;
dcSampOut = mvnrnd(theta.mudc, theta.Sigdc, N);
theta.d = dcSampOut(:,1);
theta.C = dcSampOut(:,2:end);

% linear dynamics: b, A & Q
theta.A = eye(p);
theta.b = zeros(p,1);
theta.Q = eye(p)*prior.Psi0;

if sampleDynamics
    for k = 1:p
        theta.Q(k,k)= iwishrnd(prior.Psi0, prior.nu0);
        baSamp = mvnrnd(prior.BA0, kron(theta.Q(k,k), inv(prior.Lamb0)))';
        theta.b(k) = baSamp(1);
        theta.A(k,k) = baSamp(2);
    end
end

% latent: Xori & X
% with hyper parameter: x0ori & x0
theta.Xori = zeros(p, T);
% theta.x0 =  mvnrnd(prior.mux00, prior.Sigx00)';
theta.x0 =  0*prior.mux00;
invQ0 = inv(sparse(prior.Q0));
R = chol(invQ0,'lower');
z = randn(p, 1) + R'*theta.x0;
theta.Xori(:, 1) = R'\z;
for t= 2:T
    theta.Xori(:, t) = mvnrnd(theta.b + theta.A*theta.Xori(:, t-1), theta.Q);
end

theta.X = theta.Xori - mean(theta.Xori, 2);
[QX, ~] = mgson(theta.X');
theta.X = QX';


end