
% permute observations
Lab_true = repelem(1:nClus, n);
Y_raw = Y;

idPer = randsample(N, N, false);
Lab = Lab_true(idPer)';
Y = Y_raw(idPer, :);

[~,idPer_new] = sort(Lab);
idPer_final = idPer(idPer_new);


% max cluster need
s_star = 5;

% true values

d_tmp = d(idPer);
C_tmp = C_all(idPer, :);
C_trans = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans(k, ((Lab_true(k)-1)*p+1):(Lab_true(k)*p)) = C_all(k,:);
end



x0_tmp = zeros(N*p, 1);
b_tmp = zeros(N*p, 1);
A_tmp = zeros(N*p, N*p);
Q_tmp = zeros(N*p, N*p);

latID_true = id2id(1:nClus, p);

x0_tmp(latID_true) = x0';
b_tmp(latID_true) = b;
A_tmp(latID_true, latID_true) = A;
Q_tmp(latID_true, latID_true) = Q;


XOut = zeros(N*p, T);
XOut(latID_true,:) = X;
dOut = d_tmp;
COut = C_tmp;
x0Out = x0_tmp;
bOut = b_tmp;
AOut = A_tmp;
QOut = Q_tmp;








