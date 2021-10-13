function lpdf = log_prior(theta, prior)

% to debug
% theta = ti;
% mudc: [3×1 double]
% Sigdc: [3×3 double]
% d: [30×1 double]
% C: [30×2 double]
% A: [2×2 double]
% b: [2×1 double]
% Q: [2×2 double]
% Xori: [2×1000 double]
% x0: [2×1 double]
% X: [2×1000 double]
N = length(theta.d);
p = length(theta.b);
T = size(theta.X, 2);

lpdf = 0;

% (1) mudc
lpdf = lpdf + mvnlpdf(theta.mudc, prior.deltadc0, inv(prior.Taudc0));

% (2) Sigdc
lpdf = lpdf + iwishlpdf(theta.Sigdc, prior.Psidc0, prior.nudc0);

% (3) d & C
for kk = 1:N
    lpdf = lpdf + mvnlpdf([theta.d(kk) theta.C(kk,:)]',theta.mudc, inv(theta.Sigdc));
end

% (4) Q, b & A
for kk = 1:p
    lpdf = lpdf + iwishlpdf(theta.Q(kk,kk), prior.Psi0, prior.nu0);
    baTmp = [theta.b(kk) theta.A(kk,kk)]';
    lpdf = lpdf + mvnlpdf(baTmp, prior.BA0, kron(theta.Q(kk,kk), inv(prior.Lamb0)));
end

% (6) x0
lpdf = lpdf + mvnlpdf(theta.x0, prior.mux00, inv(prior.Sigx00));

% (7) Xori
lpdf = lpdf + mvnlpdf(theta.Xori(:,1), theta.x0, inv(prior.Q0));
for t = 2:T
    lpdf = lpdf + mvnlpdf(theta.Xori(:,t), theta.b + theta.A*theta.Xori(:, t-1), inv(theta.Q));
end

end