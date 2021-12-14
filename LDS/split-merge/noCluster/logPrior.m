function lpdf = logPrior(d,X,C,b,A,Q,priortmp)

% for debug
% d = THETAstar.d;
% X = THETAstar.X;
% C = THETAstar.C;
% b = THETAstar.b;
% A = THETAstar.A;
% Q = THETAstar.Q;
% priortmp = prior(pStar);
T = length(d);
lpdf = 0;
% log-p(C)
for k = 1:size(C,1)
    lpdf = lpdf + mvnlpdf(C(k,:)', priortmp.muC0, inv(priortmp.SigC0));
end

dX = [d X]';
% log-p(d,X|b,A,Q, theta0, Q0)
lpdf = lpdf -0.5*(dX(:,1) - priortmp.theta0)'*inv(priortmp.Q0)*...
    (dX(:,1) - priortmp.theta0);
for t=1:(T-1)
   lpdf = lpdf -...
       0.5*(dX(:,t+1) - A*dX(:,t) - b)'*inv(Q)*(dX(:,t+1) - A*dX(:,t) - b); 
end

% log-p(b);log-p(A);log-p(Q)
for k = 1:length(b)
    lpdf = lpdf + mvnlpdf(b(k), priortmp.BA0(1), inv(priortmp.Lamb0(1)));
    lpdf = lpdf + mvnlpdf(A(k,k), priortmp.BA0(2), inv(priortmp.Lamb0(2,2)));
    lpdf = lpdf + log(gampdf(1/Q(k,k), priortmp.nu0/2, 2/priortmp.Psi0));
end






end