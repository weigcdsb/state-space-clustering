function [x,W,lam,lamPred] = ppasmoo_poissexp_v2(n,C,d,x0,W0,A,b,Q)

nCell = size(n, 1);
T = size(n, 2);


% Preallocate
x   = zeros(length(x0),T);
W   = zeros([size(W0) T]);
lamPred = zeros(nCell, T);

% Initialize
x(:,1)   = x0;
W(:,:,1) = W0;
lamPred(:,1)   = exp(C*x0 + d);

xpred = x;
Wpred = W;
lam = lamPred;

I = eye(size(W0));

% Forward-Pass (Filtering)
for i=2:length(n)
    xpred(:,i) = A*x(:,i-1) + b;
    lamPred(:,i) = exp(C*xpred(:,i) + d);
    Wpred(:,:,i) = A*W(:,:,i-1)*A' + Q;
    
    
    INFO = zeros(size(W0));
    SCORE = zeros(size(x0));
    
    for k=1:nCell
        INFO = INFO + C(k,:)'*(lamPred(k,i))*C(k,:);
        SCORE = SCORE + C(k,:)'*(n(k,i) - lamPred(k, i));
        
    end
    Wpostinv = inv(Wpred(:,:,i)) + INFO;
    W(:,:,i) = inv(Wpostinv);
    x(:,i)  = xpred(:,i) + W(:,:,i)*SCORE;
    
    lam(:,i) = exp(C*x(:,i) + d);
    
    [~, msgid] = lastwarn;
    if strcmp(msgid,'MATLAB:illConditionedMatrix')
        disp('error');
        lastwarn('')
        return;
    end
end
% lastwarn('')


% Backward-Pass (RTS)
for i=(T-1):-1:1
    Wi = inv(Wpred(:,:,i+1));
    J = W(:,:,i)*A'*Wi;
    x(:,i) = x(:,i) + J*(x(:,i+1) - xpred(:,i+1));
    W(:,:,i) = W(:,:,i) + J*(W(:,:,i+1)-Wpred(:,:,i+1))*J';
    lam(:,i) = exp(C*x(:,i) + d);
end


end