function gradHess = gradHessX_bt(vecX, d_tmp, C_trans_tmp, x0_tmp, Q0_tmp, Q_tmp, A_tmp, b_tmp, Y)

% for debugging...
% dX = [theta_a.d;theta_a.X];
% vecX = dX(:);
% d_tmp = zeros(N_tmp,1);
% C_trans_tmp = [ones(N_tmp,1) theta_a.C(obsIdx,:)];
% x0_tmp = prior.theta0;
% Q0_tmp = prior.Q0;
% Q_tmp = theta_a.Q;
% A_tmp = theta_a.A;
% b_tmp = theta_a.b;
% Y = Y_tmp;

T = size(Y,2);
X_all = reshape(vecX, [], T);
lam_tmp = exp(C_trans_tmp*X_all + d_tmp);

hessup = repmat((Q_tmp\A_tmp)', 1, 1, T-1);
hessub = repmat(Q_tmp\A_tmp, 1, 1, T-1);
hessmed = repmat(zeros(size(X_all, 1)),1,1,T);
for t = 1:T
    if (t==1)
        hessmed(:,:,t) = -C_trans_tmp'*diag(lam_tmp(:,t))*C_trans_tmp-...
            inv(Q0_tmp)- A_tmp'*(Q_tmp\A_tmp);
    elseif (t == T)
        hessmed(:,:,t) = -C_trans_tmp'*diag(lam_tmp(:,t))*C_trans_tmp -inv(Q_tmp);
    else
        hessmed(:,:,t) = -C_trans_tmp'*diag(lam_tmp(:,t))*C_trans_tmp -...
            inv(Q_tmp) - A_tmp'*(Q_tmp\A_tmp);
    end
end

gradHess{1} = C_trans_tmp'*(Y - lam_tmp) + [-Q0_tmp\(X_all(:,1) - x0_tmp)+...
    A_tmp'*(Q_tmp\(X_all(:,2) - A_tmp*X_all(:,1)-b_tmp(:,1))),...
    -Q_tmp\(X_all(:,2:(T-1)) - A_tmp*X_all(:,1:(T-2))-b_tmp(:, 1:(T-2)))+...
    A_tmp'*(Q_tmp\(X_all(:,3:T) - A_tmp*X_all(:,2:(T-1))-b_tmp(:,2:(T-1)))),...
    -Q_tmp\(X_all(:,T) - A_tmp*X_all(:,T-1)-b_tmp(:,T-1))];
gradHess{1} = gradHess{1}(:);

gradHess{2} = blktridiag(hessmed,hessub,hessup);
% gradHess{2} = (gradHess{2} + gradHess{2}')/2;


end