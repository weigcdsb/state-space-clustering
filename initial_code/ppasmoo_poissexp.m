% Point-process adaptive smoothing w/ Poisson likelihood (log-link)
%  filtering via Eden et al. Neural Comp 2004
%  then a backward pass based on Rauch-Tung-Striebel

function [b,W,lam] = ppasmoo_poissexp(n,X,b0,W0,F,Q,offset)

nCell = size(n, 1);
T = size(n, 2);

if nargin<7, offset=zeros(nCell, T); end

% Preallocate
b   = zeros(length(b0),T);
W   = zeros([size(W0) length(n)]);
lam = zeros(nCell, T);

% Initialize
b(:,1)   = b0;
W(:,:,1) = W0;
lam(:,1)   = exp(X(:,1)*b0 + offset(:,1));

bpred = b;
Wpred = W;

I = eye(size(W0));

% Forward-Pass (Filtering)
for i=2:length(n)
    bpred(:,i) = F*b(:,i-1);
    lam(:,i) = exp(X(:,i)*bpred(:,i) + offset(:,i));
    Wpred(:,:,i) = F*W(:,:,i-1)*F' + Q;
    
    
    INFO = zeros(size(W0));
    SCORE = zeros(size(b0));
    
    for k=1:nCell
        INFO = INFO + X(k,i)'*(lam(k,i))*X(k,i);
        SCORE = SCORE + X(k,i)'*(n(k,i) - lam(k, i));
        
    end
    Wpostinv = inv(Wpred(:,:,i)) + INFO;
    W(:,:,i) = inv(Wpostinv);
    b(:,i)  = bpred(:,i) + W(:,:,i)*SCORE;
    
    [~, msgid] = lastwarn;
    if strcmp(msgid,'MATLAB:illConditionedMatrix')
        return;
    end
end


% Backward-Pass (RTS)
for i=(T-1):-1:1
    Wi = inv(Wpred(:,:,i+1));
    Fsquig = inv(F)*(I-Q*Wi);
    Ksquig = inv(F)*Q*Wi;
    
    b(:,i)=Fsquig*b(:,i+1) + Ksquig*bpred(:,i+1);
    C = W(:,:,i)*F'*Wi;
    W(:,:,i) = W(:,:,i) + C*(W(:,:,i+1)-Wpred(:,:,i+1))*C';
end


end

