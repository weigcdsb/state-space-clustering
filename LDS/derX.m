function der = derX(vecX, d_tmp, C_trans_tmp, x0_tmp, Q0_tmp, Q_tmp, A_tmp, b_tmp, Y)

T = size(Y,2);
X_all = reshape(vecX, [], T);
lam_tmp = exp(C_trans_tmp*X_all + d_tmp);

derMat = C_trans_tmp'*(Y - lam_tmp) + [-Q0_tmp\(X_all(:,1) - x0_tmp)+...
    A_tmp'*(Q_tmp\(X_all(:,2) - A_tmp*X_all(:,1)-b_tmp)),...
    -Q_tmp\(X_all(:,2:(T-1)) - A_tmp*X_all(:,1:(T-2))-b_tmp)+...
    A_tmp'*(Q_tmp\(X_all(:,3:T) - A_tmp*X_all(:,2:(T-1))-b_tmp)),...
    -Q_tmp\(X_all(:,T) - A_tmp*X_all(:,T-1)-b_tmp)];

der = derMat(:);


end