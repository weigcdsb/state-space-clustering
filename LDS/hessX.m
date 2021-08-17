function hess = hessX(vecX, d_tmp, C_trans_tmp, Q0_tmp, Q_tmp, A_tmp, Y)

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
    
hess = blktridiag(hessmed,hessub,hessup);

end
