function lpdf = dKlpdf_blk_nod(Y, K, X, muK, sig2K)

n = size(K, 1);
p = size(K, 2);
CTest = K/(sqrtm(K'*K));

logLam = CTest*X;
lpdf = sum(-exp(logLam) + Y.*logLam, 'all') -...
    (1/2)*sum(((K(:) - muK).^2)./sig2K);

end