

X_fit = repmat(X, 1, 1, ng); % true
d_fit = repmat(d,1,ng); % true
C_fit = repmat(C_all, 1,1,ng); % true
x0_fit = repmat([mu1 mu2 mu3]',1,ng); %true
Q0_fit = repmat(Q0,1,1,ng); % true
A_fit = repmat(A,1,1,ng); % true
b_fit = repmat(b,1,ng); % true
Q_fit = repmat(Q,1,1,ng); % true

X_fit = zeros(nClus*p, T, ng);
d_fit = zeros(n*nClus, ng);
C_fit = zeros(n*nClus, p, ng);
x0_fit = zeros(nClus*p, ng);
Q0_fit = zeros(p, p, ng);
A_fit = zeros(nClus*p, nClus*p, ng);
b_fit = zeros(nClus*p, ng);
Q_fit = zeros(nClus*p, nClus*p, ng);

