function lpdf = dKlpdf_blk2(Y, d, K, X,mud, sig2d, muK, SigK)

% for debugging...
% dKvec = [d_fit(Lab == l,l, g-1); reshape(K_fit(Lab == l,latentId,g-1), [], 1)];
% Ytest = Y(Lab == l,:);
% dtest = dKvec(1:nj);
% Ktest = reshape(dKvec((nj+1):end), [], p);
% Xtest = X_fit(latentId,:,g);
% mudtest = mud_fit(l,g-1);
% sig2dtest = sig2d_fit(l,g-1);
% muKtest = muK_fit(:,l,g-1);
% SigKtest = SigK_fit(:,:,l,g-1);

n = size(K, 1);
p = size(K, 2);
CTest = K/(sqrtm(K'*K));

logLam = d + CTest*X;
KDeviate = K - repmat(muK', n, 1);

lpdf = sum(-exp(logLam) + Y.*logLam, 'all') -...
    (1/2)*sum(((d - mud).^2)./sig2d) -...
    (1/2)*(log(det(SigK)) + trace((KDeviate/(SigK))*KDeviate'));


end