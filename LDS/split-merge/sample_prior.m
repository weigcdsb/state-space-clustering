function [dOut, COut, XOut, x0Out, mudcOut, SigdcOut,...
    QOut, AOut, bOut] =...
    sample_prior(Q0_f, mux00_f, Sigx00_f, deltadc0, Taudc0,Psidc0,nudc0,...
    Psi0,nu0, N, T, p)

AOut = eye(p);
bOut = zeros(p,1);
QOut = eye(p);
XOut = zeros(p, T);

mudcOuter = mvnrnd(deltadc0, Taudc0);
SigdcOuter = iwishrnd(Psidc0,nudc0);
mudcOut = mudcOuter';
SigdcOut = SigdcOuter;
dcSampOut = mvnrnd(mudcOut, SigdcOut, N);
dOut = dcSampOut(:,1);
COut = dcSampOut(:,2:end);

x0Out =  mvnrnd(mux00_f(1), Sigx00_f(1))';
for k =1:p
    QOut(k,k) = iwishrnd(Psi0,nu0);
end
invQ0 = inv(sparse(Q0_f(1)));
R = chol(invQ0,'lower');
z = randn(p, 1) + R'*x0Out;
XOut(:, 1) = R'\z;

for t= 2:T
    XOut(:, t) = mvnrnd(eye(p)*XOut(:, t-1), QOut);
end

XOut = XOut - min(XOut,[], 2);
XOut= (diag(range(XOut,2)))\XOut;

end