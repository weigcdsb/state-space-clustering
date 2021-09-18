function [lpdf,glpdf] = logpdfX(vecX, d_tmp, C_trans_tmp, x0_tmp,...
    Q0_tmp, Q_tmp, A_tmp, b_tmp, Y)

glpdf = derX(vecX, d_tmp, C_trans_tmp, x0_tmp, Q0_tmp, Q_tmp, A_tmp, b_tmp, Y);
logNPrior = @(X) -1/2*(X(:,1) - x0_tmp)'*inv(Q0_tmp)*(X(:,1) - x0_tmp) -...
            1/2*trace((X(:,2:end) - A_tmp*X(:,1:(end-1)) - b_tmp)'*inv(Q_tmp)*...
            (X(:,2:end) - A_tmp*X(:,1:(end-1)) - b_tmp));
        
lamX = @(X) exp(C_trans_tmp*X + d_tmp) ;
T = size(Y, 2);
lpdf = sum(log(poisspdf(Y, lamX(reshape(vecX, [], T)))), 'all') +...
    logNPrior(reshape(vecX, [], T));

end







