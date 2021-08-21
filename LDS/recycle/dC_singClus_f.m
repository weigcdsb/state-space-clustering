function [dOut, COut, llhd_trace] = dC_singClus_f(Y,dMar,CMar,x0Mar,Q0Mar,bMar,AMar,QMar,...
    Sigx00Mar,mux00Mar,mudc0,Sigdc0,SigbA0Mar,mubA0Mar,Psi0,nu0,... % priors
    iter, tol) 

N = size(Y,1);
T = size(Y,2);
p = length(x0Mar);

XMar = ppasmoo_poissexp_v2(Y, CMar, dMar, x0Mar, Q0Mar, AMar, bMar, QMar);
llhd_trace = zeros(1,iter);

lam_tmp = exp(CMar*XMar + dMar);
llhd_trace(1) = sum(sum(log(poisspdf(Y, lam_tmp)), 2));


for i = 1:iter
    
    % (1) update XMar
    XMar = ppasmoo_poissexp_v2(Y, CMar, dMar, x0Mar, Q0Mar, AMar, bMar, QMar);
    
    % (2) update x0Mar
    invSigx0 = sparse(inv(Sigx00Mar) + inv(Q0Mar));
    x0Mar = invSigx0\(Sigx00Mar\mux00Mar + Q0Mar\XMar(:,1));
    
    % (3) update dMar & CMar
    % Laplace approximation
    dMarPre = dMar;
    CMarPre = CMar;
    
    for j = 1:N
        X_tmpdc = [ones(1, T) ; XMar]';
        lamdc = @(dc) exp(X_tmpdc*dc);
        
        derdc = @(dc) X_tmpdc'*(Y(j,:)' - lamdc(dc)) - Sigdc0\(dc - mudc0);
        hessdc = @(dc) -X_tmpdc'*diag(lamdc(dc))*X_tmpdc - inv(Sigdc0);
        [mudc,~,~,~] = newton(derdc,hessdc,...
            [dMar(j) CMar(j,:)]',1e-8,1000);
        
        dMar(j) = mudc(1);
        CMar(j,:) = mudc(2:end);
    end
    
    % (4) update bMar & AMar
    Xpost_tmp = XMar(:, 2:T);
    Xpost_tmp2 = Xpost_tmp(:);
    
    XbA_tmp = kron([ones(1,T-1); XMar(:, 1:(T-1))]', eye(p));
    invSigbA_tmp = sparse(inv(SigbA0Mar) +...
        XbA_tmp'*kron(eye(T-1), inv(QMar))*XbA_tmp);
    mubA_tmp = invSigbA_tmp\(SigbA0Mar\mubA0Mar +...
        XbA_tmp'*kron(eye(T-1), inv(QMar))*Xpost_tmp2);
    bA_tmp = reshape(mubA_tmp, p, []);
    bMar = bA_tmp(:,1);
    AMar = bA_tmp(:,2:end);
    
    % (5) update QMar
    mux = AMar*XMar(:, 1:(T-1)) + bMar;
    xq = XMar(:, 2:T) -mux;
    PsiQ = Psi0 + xq*xq';
    nuQ = T-1 + nu0;
    QMar = PsiQ/(nuQ + p + 1);
    
    lam_tmp = exp(CMar*XMar + dMar);
    llhd_trace(i) = sum(sum(log(poisspdf(Y, lam_tmp)), 2));

    disp(norm([dMarPre CMarPre] - [dMar CMar])/norm([dMarPre CMarPre]))
    if(norm([dMarPre CMarPre] - [dMar CMar])/norm([dMarPre CMarPre]) < tol)
        break
    end
    
end


dOut = dMar;
COut = CMar;



end