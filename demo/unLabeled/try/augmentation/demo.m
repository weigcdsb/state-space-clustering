addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
% addpath(genpath('D:\github\state-space-clustering'));

%% simulation
rng(1)
n = 10;
nClus = 3;
N = n*nClus;
p = 2;
T = 1000;

Lab = repelem(1:nClus, n);
pLab = repelem(1:nClus, p);

d = randn(n*nClus,1)*0;
d(1:10) = d(1:10) + 1;
C_trans = zeros(n*nClus, p*nClus);
for k = 1:length(Lab)
    C_trans(k, ((Lab(k)-1)*p+1):(Lab(k)*p)) =...
        sum(Lab(1:k)==Lab(k))/sum(Lab==Lab(k))+[-1 2];
end

X = zeros(p*nClus, T);
x0 = zeros(p*nClus, 1);
Q0 = eye(nClus*p);
X(:,1) = mvnrnd(x0, Q0)';

b1 = randn(p, 1)*1e-3;
b2 = randn(p, 1)*1e-3;
b3 = randn(p, 1)*1e-4;
b = [b1;b2;b3];

Q1 = 1e-3*eye(p);
Q2 = 1e-4*eye(p);
Q3 = 1e-5*eye(p);
Q = blkdiag(Q1, Q2, Q3);

%
A = eye(size(Q,1));
while any(imag(eig(A))==0)
    A= randn(size(Q));
    A = A-diag(diag(A));
    A(squareform(pdist(pLab'))==0)=0;
    A = A./sqrt(sum((A-diag(diag(A))).^2,2))*0.1;
    %     A = A+eye(size(Q,1))*0.92;
    A = A + diag(randn(nClus*p, 1)*1e-1 + 0.92);
end

% let's generate lambda
logLam = zeros(n*nClus, T);
logLam(:,1) = d + C_trans*X(:,1);

for t=2:T
    X(:, t) = mvnrnd(A*X(:, t-1) + b, Q)';
    logLam(:, t) = d + C_trans*X(:,t);
end

Y = poissrnd(exp(logLam));

gtmp = -mean(X, 2);
Mdiag = {};
for l = unique(Lab)
    latidTmp = id2id(l,p);
    [QX, RX] = mgson(X(latidTmp,:)');
    Mdiag{l} = inv(RX');
end
M = sparse(blkdiag(Mdiag{:}));
g = M*gtmp;

X = M*X + g;
x0 = M*x0 + g;
d = d - (C_trans/M)*g;
C_trans = C_trans/M;
A = M*A/M;
b = M*b + g - (M*A/M)*g;
Q = M*Q*M';
Q0 = M*Q0*M';

clusterPlot(Y, Lab)
%%
% Y is the only observation...
% some functions
lAbsGam = @(x) log(abs(gamma(x)));

%% MCMC settings
rng(1)
p=1;
ng = 100;
t_max = N;

% this is the DP setting, replace to MFM later...
% DPMM = true;
% alpha_random = true;
% sigma_alpha = 0.1; % scale for MH proposals in alpha move
% alphaDP = 1;
% log_v = (1:t_max+1)*log(alphaDP) - lAbsGam(alphaDP+N) + lAbsGam(alphaDP);
% a = 1;
% b = 0;


% MFM:
DPMM = false;
alpha_random = false;
MFMgamma = 1;
% K ~ Geometric(r)
r = 0.3;
log_pk = @(k) log(r) + (k-1)*log(1-r);

a = MFMgamma;
b = MFMgamma;
log_v = MFMcoeff(log_pk, MFMgamma, N, t_max + 1);

logNb = log((1:N) + b);

% priors...
% theta = (d x)
prior.theta0 = zeros(1+p,1);
prior.Q0 = eye(1+p);

prior.muC0 = zeros(p,1);
prior.SigC0 = eye(p);

prior.BA0 =[0 1]';
prior.Lamb0 = eye(2);
prior.Psi0 = 1e-3;
prior.nu0 = 1+2;

% pre-allocation
t_fit = zeros(1, ng);
Z_fit = zeros(N, ng);
numClus_fit = zeros(t_max + 3, ng);

t_fit(1) = 1;
Z_fit(:,1) = ones(N, 1);
numClus_fit(1,1) = N;
actList = zeros(t_max+3,1); actList(1) = 1;
c_next = 2;

for k = 1:t_fit(1)
    THETA{1}(k) = sample_prior_new(prior, N, T, p, true, Inf);
end

%% MCMC

burnIn = round(ng/10);
epsilon = 0.01*ones(N,1);

for i = 1:N
    optdc.M=1;
    optdc.Madapt=0;
    OPTDC{i} = optdc;
    OPTDC{i}.epsilon = epsilon(i); % no tune of epsilon
end

simMat = zeros(N,N);
for k = 1:size(simMat, 1)
    simMat(k,:) = simMat(k,:) + (Z_fit(k, 1) == Z_fit(:, 1))';
end
fitMFRTrace = zeros(N, T, ng);
for g = 2:ng
    
    if(g < burnIn);disp("iter " + g + ", changing"); % change epsilon
    elseif(g == burnIn) % tune epsilon
        %         for(k = 1:N);OPTDC{k}.Madapt=10;end
        disp("iter " + g + ", tuning");
    else;disp("iter " + g + ", tuned");
    end % fix epsilon
    
    % (1) update cluster index z_i
    Z_fit(:,g) = Z_fit(:,g-1);
    numClus_fit(:,g) = numClus_fit(:,g-1);
    t_fit(g) = t_fit(g-1);
    
    if DPMM && alpha_random
        % MH move for DP concentration parameter (using p_alpha(a) = exp(-a) = Exp(a|1))
        aprop = alphaDP*exp(rand*sigma_alpha);
        top = t_fit(g)*log(aprop) - lAbsGam(aprop+N) + lAbsGam(aprop) - aprop + log(aprop);
        bot = t_fit(g)*log(alphaDP) - lAbsGam(alphaDP+N) +...
            lAbsGam(alphaDP) - alphaDP + log(alphaDP);
        if rand < min(1, exp(top-bot))
            alphaDP = aprop;
        end
        log_v = (1:t_max+1)*log(alphaDP) - lAbsGam(alphaDP+N) + lAbsGam(alphaDP);
        disp(alphaDP)
    end
    
    
    for ii = 1:N
        
        % (a) remove point ii from its cluster
        c = Z_fit(ii, g);
        numClus_fit(c,g) = numClus_fit(c,g) - 1;
        if(numClus_fit(c,g) > 0)
            c_prop = c_next;
            THETA{g-1}(c_prop) = sample_prior_new(prior, N, T, p, false, Inf);
        else
            c_prop = c;
            actList = ordered_remove(c, actList, t_fit(g));
            t_fit(g) = t_fit(g) - 1;
        end
        
        %(b) compute probabilities for resampling
        log_p = zeros(t_fit(g)+1,1);
        for j = 1:t_fit(g)
            cc = actList(j);
            logLike = nansum(log(poisspdf(Y(ii,:)',...
                exp(THETA{g-1}(cc).d' + THETA{g-1}(cc).X'* THETA{g-1}(cc).C(ii,:)))));
            log_p(j) = logNb(numClus_fit(cc,g)) + logLike;
        end
        
        logLike = nansum(log(poisspdf(Y(ii,:)',...
            exp(THETA{g-1}(c_prop).d' + THETA{g-1}(c_prop).X'* THETA{g-1}(c_prop).C(ii,:)))));
        log_p(t_fit(g)+1) = log_v(t_fit(g)+1)-log_v(t_fit(g)) +...
            log(a) + logLike;
        
        % (c) sample a new cluster for it
        j = randlogp(log_p, t_fit(g)+1);
        
        % (d) add point i to its new clusters
        if j <= t_fit(g)
            c = actList(j);
        else
            c = c_prop;
            actList = ordered_insert(c, actList, t_fit(g));
            t_fit(g) = t_fit(g) + 1;
            c_next = ordered_next(actList);
        end
        
        Z_fit(ii,g) = c;
        numClus_fit(c,g) = numClus_fit(c,g) + 1;
    end
    
    % (2) update augmented c_i
    THETA{g} = THETA{g-1};
    for i = 1:N
        for j = 1:t_fit(g)
            ctmp = actList(j);
            X_tmpC = THETA{g}(ctmp).X';
            lamC = @(c) exp(THETA{g}(ctmp).d' + X_tmpC*c);
            
            % use NUTS
            lpdf = @(c) sum(log(poisspdf(Y(i,:)', lamC(c)))) +...
                log(mvnpdf(c, prior.muC0, prior.SigC0));
            glpdf = @(c) X_tmpC'*(Y(i,:)' - lamC(c)) - prior.SigC0\(c - prior.muC0);
            
            fg=@(dc_r) deal(lpdf(dc_r'), glpdf(dc_r')'); % log density and gradient
            c0 = THETA{g-1}(ctmp).C(i,:)';
            
            [c_NUTS, ~, diagn]=hmc_nuts(fg, c0',OPTDC{i});
            epsilon(i) = diagn.opt.epsilonbar;
            THETA{g}(ctmp).C(i,:) = c_NUTS(end,:);
            
            
            % ues MH
%             derc = @(c) X_tmpC'*(Y(i,:)' - lamC(c)) - prior.SigC0\(c - prior.muC0);
%             hessc = @(c) -X_tmpC'*diag(lamC(c))*X_tmpC - inv(prior.SigC0);
%             c0 = THETA{g-1}(ctmp).C(i,:)';
%             
%             [muc,~,niSigc,~] = newton(derc,hessc,c0,1e-8,1000);
%             if(sum(isnan(muc)) ~= 0)
%                 disp('use 0')
%                 [muc,~,niSigc,~] = newton(derc,hessc,zeros(size(c0)),1e-8,1000);
%             end
            
%             R = chol(-niSigc,'lower'); % sparse
%             z = randn(length(c0), 1) + R'*c0;
%             cStar = R'\z;
%             
%             % lhr
%             lhr = sum(log(poisspdf(Y(i,:)', lamC(cStar)))) -...
%                 sum(log(poisspdf(Y(i,:)', lamC(c0)))) +...
%                 log(mvnpdf(cStar, prior.muC0, prior.SigC0)) -...
%                 log(mvnpdf(c0, prior.muC0, prior.SigC0));
%             
%             if(log(rand(1)) < lhr)
%                 THETA{g-1}(ctmp).C(i,:) = cStar;
%             else
%                 THETA{g-1}(ctmp).C(i,:) = c0;
%             end
            
            
            % normal approximation
%             Rc = chol(-niSigc,'lower'); % sparse
%             zc = randn(length(muc), 1) + Rc'*muc;
%             THETA{g}(ctmp).C(i,:) = Rc'\zc;
            
        end
    end
    
    %     if(g == burnIn)
    %         for i = 1:N
    %             for j = 1:t_fit(g)
    %                 OPTDC{i,j}.Madapt=0;OPTDC{k}.epsilon = epsilon(k);
    %             end
    %         end
    %     end
    
    
    % (3) upate the parameters
    
    for j = 1:t_fit(g)
        c = actList(j);
        obsIdx = find(Z_fit(:,g) == c);
        
        THETA{g}(c) =...
            update_cluster_new_aug(Y(obsIdx,:),THETA{g}(c),THETA{g}(c),...
            prior, T, p, obsIdx);
    end
    
    figure(1)
    clusterPlot(Y, Z_fit(:,g)')
    
    figure(2)
    for k = 1:size(simMat, 1)
        simMat(k,:) = simMat(k,:) + (Z_fit(k, g) == Z_fit(:, g))';
    end
    
    imagesc(simMat/g)
    colormap(flipud(hot))
    colorbar()
    
    
    figure(3)
    subplot(1,2,1)
    imagesc(Y)
    cLim = caxis;
    title('true')
    colorbar()
    subplot(1,2,2)
    fitMFR = zeros(N, T);
    for k  = 1:N
        fitMFR(k,:) = exp([1 THETA{g}(Z_fit(k,g)).C(k,:)]*...
            [THETA{g}(Z_fit(k,g)).d ;THETA{g}(Z_fit(k,g)).X]);
    end
    fitMFRTrace(:,:,g) = fitMFR;
    imagesc(fitMFR)
    set(gca,'CLim',cLim)
    colorbar()
    title('fit')
    
    figure(4)
    plot(t_fit(1:g))
    
end

%%
figure(1)
plot(t_fit)
title('number of cluster')

figure(2)
zMax = max(Z_fit(:));
Z_trace = Z_fit;
Z_trace2 = Z_trace + 0.2*rand(N,1);
hold on
for k = 1:N
    p(k)=plot(Z_trace2(k,:));
    if k < n+1
        set(p(k),'Color', 'r');
    elseif k < 2*n+1
        set(p(k),'Color', 'g');
    else
        set(p(k),'Color', 'b');
    end
end
hold off
ylim([0 zMax+1])
yticks(1:zMax)
% xlim([0 10])
clusLab = [];
for c = 1:zMax
    clusLab{c} = 'cluster ' + string(c);
end
yticklabels(clusLab)
title('cluster trace for each neuron')


figure(3)
imagesc(exp(C_trans*X + d))
cLim = caxis;
title('true')
colorbar()

figure(4)
idx = round(g/2):g;
simMat = zeros(N,N);
for g = idx
    for k = 1:size(simMat, 1)
        simMat(k,:) = simMat(k,:) + (Z_fit(k, g) == Z_fit(:, g))';
    end
end

imagesc(simMat/length(idx))
colormap(flipud(hot))
colorbar()

% histogram of cluster number
figure(5)
histogram(t_fit(idx),'Normalization','probability')




