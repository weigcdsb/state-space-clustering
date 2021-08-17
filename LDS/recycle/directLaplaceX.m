g = 2;
disp(g)

% direct Laplace approximation
d_tmp = d_fit(:,g-1);
C_trans_tmp = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans_tmp(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) = C_fit(k,:,g-1);
end

X_all = X_fit(:,:,g-1);
vecX = X_all(:);

x0_tmp = x0_fit(:,g-1);
Q0_tmp = Q0_fit(:,:,g-1);
Q_tmp = Q_fit(:,:,g-1);
A_tmp = A_fit(:,:,g-1);
b_tmp = b_fit(:,g-1);

der = @(vecX) derX(vecX, d_tmp, C_trans_tmp, x0_tmp, Q0_tmp, Q_tmp, A_tmp, b_tmp, Y);
hess = @(vecX) hessX(vecX, d_tmp, C_trans_tmp, x0_tmp, Q0_tmp, Q_tmp, A_tmp, b_tmp, Y);

[muX,fx,niSigX,xx] = newton(der,hess,...
    X_all(:),1e-10,1000);

SigX = -inv(niSigX);
SigX = (SigX + SigX')/2;
muX2 = mvnrnd(muX, SigX);


subplot(1,2,1)
plot(reshape(muX, [], T)')
subplot(1,2,2)
plot(X')

f = der;
df = hess;
x0 = X_all(:);
TolX = 1e-8;
MaxIter = 1000;


