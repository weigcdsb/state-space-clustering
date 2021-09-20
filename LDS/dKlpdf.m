function lpdf = dKlpdf(Y, d_tmp, K, X, Lab, mud, sig2d)

N = size(K, 1);
p = size(K, 2);
nClus = size(X,1)/p;
C = K/(sqrtm(K'*K));

C_tmp = zeros(N, p*nClus);
mud_tmp = zeros(N,1);
sig2d_tmp = zeros(N,1);
for j = 1:length(Lab)
    C_tmp(j, id2id(Lab(j), p)) = C(j,:);
    mud_tmp(j) = mud(Lab(j));
    sig2d_tmp(j) = sig2d(Lab(j));
end

logLam = d_tmp + C_tmp*X;
lpdf = sum(-exp(logLam) + Y.*logLam, 'all') -...
    (1/2)*sum(((d_tmp - mud_tmp).^2)./sig2d_tmp) - trace(K'*K/2);

end