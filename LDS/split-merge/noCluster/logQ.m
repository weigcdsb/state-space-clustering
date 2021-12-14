function logQout = logQ(d,X,C,b,A,Q,Qtrantmp)

% for debug
% d = THETAstar.d;
% X = THETAstar.X;
% C = THETAstar.C;
% b = THETAstar.b;
% A = THETAstar.A;
% Q = THETAstar.Q;
% Qtrantmp = Qtran{pStar}(j);

% d = THETA{g-1}(j).d';
% X = THETA{g-1}(j).X';
% C = THETA{g-1}(j).C;
% b = THETA{g-1}(j).b;
% A = THETA{g-1}(j).A;
% Q = THETA{g-1}(j).Q;
% Qtrantmp = Qtran{pFit(j,g-1)}(j);


logQout = 0;
T = length(d);
% log-p(C)
for k = 1:size(C,2)
    logQout = logQout + mvnlpdf(C(:,k), Qtrantmp.MC(:,k), inv(0.81*Qtrantmp.VC(:,:,k)));
end


dX = [d X]';
% log-p(d,X|b,A,Q, theta0, Q0)
logQout = logQout -0.5*(d(1) - Qtrantmp.Md(1))'*inv(0.81*Qtrantmp.Vd(1,1))*...
    (d(1) - Qtrantmp.Md(1));
for k = 1:size(C,2)
    logQout = logQout -0.5*(X(1,k) - Qtrantmp.MX(1,k))'*inv(0.81*Qtrantmp.VX(1,1,k))*...
        (X(1,k) - Qtrantmp.MX(1,k));
end
for t=1:(T-1)
   logQout = logQout -...
       0.5*(dX(:,t+1) - A*dX(:,t) - b)'*inv(Q)*(dX(:,t+1) - A*dX(:,t) - b); 
end

for k = 1:length(b)
    logQout = logQout + mvnlpdf(b(k), Qtrantmp.Mb(k), inv(0.81*Qtrantmp.Vb(k,k)));
    logQout = logQout + mvnlpdf(A(k,k),Qtrantmp.MA(k), inv(0.81*Qtrantmp.VA(k,k)));
    logQout = logQout + log(gampdf(1/Q(k,k), 1.6, 1/(1.6*Qtrantmp.MQ(k))));
end





end