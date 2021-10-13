function [Z, actList, numClus_fit_tmp, t_fit_tmp, theta] =...
    splitMerge(Y, Z, zs, S, theta, actList, N, T, p,...
    numClus_fit_tmp, t_fit_tmp, prior, a, b, log_v, n_split, n_merge, OPTDC)

% to debug
% Z = Z_fit(:,g);
% theta = THETA{g};
% numClus_fit_tmp = numClus_fit(:,g);
% t_fit_tmp = t_fit(g);

lAbsGam = @(x) log(abs(gamma(x)));

% (a) randomly choose a pair of indices
rdIdx = randsample(N,2);
ism = rdIdx(1);
jsm = rdIdx(2);

ci0 = Z(ism);
cj0 = Z(jsm);
ti0 = theta(ci0);
tj0 = theta(cj0);

%     if(ci0 == cj0)
%         ti0 = sample_prior2(prior, N, T, p, false);
%         preIdx = find(Z == ci0);
%         for s= 1:n_split
%             [ti0, ~, ~] =...
%                 update_cluster(Y(preIdx,:),ti0,ti0,...
%                 prior, N, T, p, preIdx, true, false, OPTDC(preIdx));
%         end
%     else
%         ti0 = sample_prior2(prior, N, T, p, false);
%         tj0 = sample_prior2(prior, N, T, p, false);
%         preIdx_i = find(Z == ci0);
%         preIdx_j = find(Z == cj0);
%         for m = 1:n_merge
%             [ti0, ~, ~] =...
%                 update_cluster(Y(preIdx_i,:),ti0,ti0,...
%                 prior, N, T, p, preIdx_i, true, false, OPTDC(preIdx_i));
%             [tj0, ~, ~] =...
%                 update_cluster(Y(preIdx_j,:),tj0,tj0,...
%                 prior, N, T, p, preIdx_j, true, false, OPTDC(preIdx_j));
%         end
%     end

% (b) set S(1),...,S(ns) to the indices of
% the points in clusters ci0 and cj0
ns = 0;
for k = 1:N
    if(Z(k) == ci0 || Z(k) == cj0)
        ns = ns + 1;
        S(ns) = k;
    end
end

% (c) find available cluster IDs
% for merge and split parameters
k = 1;
while(actList(k) == k); k = k+1;end;cm = k;
while(actList(k) == k+1); k = k+1;end;ci = k+1;
while(actList(k) == k+2); k = k+1;end;cj = k+2;
tm = theta(cm);
ti = theta(ci);
tj = theta(cj);

% (d) randomly choose the merge launch state
tm = sample_prior2(prior, N, T, p, false); % ti0
mIdx = S(1:ns);
for m = 1:n_merge
    [tm, ~, ~] =...
        update_cluster(Y(mIdx,:),tm,tm,...
        prior, N, T, p, mIdx, true, false, OPTDC(mIdx));
end


% (e) randomly choose the split lauch state
ti = sample_prior2(prior, N, T, p, false); % ti0
tj = sample_prior2(prior, N, T, p, false); % tj0
zs(ism) = ci;
zs(jsm) = cj;

kOut = setdiff(S(1:ns), [ism,jsm]);
splitIdx = binornd(1,.5,length(kOut), 1);
siIdx = kOut(splitIdx == 1);
sjIdx = kOut(splitIdx == 0);
zs(siIdx) = ci;
zs(sjIdx) = cj;
iIdx = [ism;siIdx];
jIdx = [jsm;sjIdx];
ni = length(iIdx);
nj = length(jIdx);

% make several moves (restricted Gibbs)
for s = 1:n_split
    [zs,ti,tj,ci,cj,log_p, ni, nj, iIdx, jIdx] =...
        restricted_gibbs(zs,zs,ti,ti,tj,tj,ci,ci,cj,cj,...
        iIdx,jIdx, ni,nj,ism,jsm,S,ns,Y,b,prior, N, T, p,true,OPTDC);
    
    %         figure(2)
    %         clusterPlot(Y(1:20,:), zs(1:20)')
end

% (f) make proposal
if ci0 == cj0 % propose a split
    % make one final sweep and compute its density
    [zs,ti,tj,ci,cj,log_prop_ab, ni, nj] =...
        restricted_gibbs(zs,zs,ti,ti,tj,tj,ci,ci,cj,cj,...
        iIdx,jIdx, ni,nj,ism,jsm,S,ns,Y,b,prior, N, T, p,true,OPTDC);
    
    % compute density of Lmerge to original state
    [ti0, ~, log_prop_ba] =...
        update_cluster(Y(mIdx,:),tm,ti0,...
        prior, N, T, p, mIdx, false, true, OPTDC(mIdx));
    
    % compute acceptance probability
    log_prior_b = log_v(t_fit_tmp+1) + lAbsGam(ni+b) +...
        lAbsGam(nj+b) - 2*lAbsGam(a) + log_prior(ti, prior);
    log_prior_a = log_v(t_fit_tmp) + lAbsGam(ns+b) - lAbsGam(a) + log_prior(ti0, prior);
    llhd_ratio = 0;
    for ks = 1:ns
        k = S(ks);
        if zs(k) == ci
            lamTmp = exp(ti.C(k,:)*ti.X + ti.d(k));
        else
            lamTmp = exp(tj.C(k,:)*tj.X + tj.d(k));
        end
        
        lamTmp0 = exp(ti0.C(k,:)*ti0.X + ti0.d(k));
        llhd_ratio = llhd_ratio +...
            sum(log(poisspdf(Y(k,:), lamTmp))) -...
            sum(log(poisspdf(Y(k,:), lamTmp0)));
    end
    p_accept = min(1, exp(log_prop_ba-log_prop_ab +...
        log_prior_b-log_prior_a + llhd_ratio));
    
    if rand < p_accept % accept split
        disp('accept split')
        for ks = 1:ns
            Z(S(ks)) = zs(S(ks));
        end
        actList = ordered_remove(ci0, actList, t_fit_tmp);
        actList = ordered_insert(ci, actList, t_fit_tmp-1);
        actList = ordered_insert(cj, actList, t_fit_tmp);
        
        numClus_fit_tmp(ci0) = 0;
        numClus_fit_tmp(ci) = ni;
        numClus_fit_tmp(cj) = nj;
        t_fit_tmp = t_fit_tmp + 1;
        
        theta(ci) = ti;
        theta(cj) = tj;
    end
else % propose a merge
    % make one final sweep and compute its probability density
    [tm, ~, log_prop_ab] =...
        update_cluster(Y(mIdx,:),tm,tm,...
        prior, N, T, p, mIdx, true, true, OPTDC(mIdx));
    
    % compute probability density of going from split launch state to original state
    [~,~,~,~,~,log_prop_ba, ni, nj, iIdx, jIdx] =...
        restricted_gibbs(zs,Z,ti,ti0,tj,tj0,ci,ci0,cj,cj0,...
        iIdx,jIdx, ni,nj,ism,jsm,S,ns,Y,b,prior, N, T, p,false,OPTDC);
    
    % compute acceptance probability
    log_prior_b = log_v(t_fit_tmp-1) + lAbsGam(ns+b)-lAbsGam(a) + log_prior(tm, prior);
    log_prior_a = log_v(t_fit_tmp) + lAbsGam(ni+b)+lAbsGam(nj+b)-2*lAbsGam(a) +...
        log_prior(ti0, prior) + log_prior(tj0, prior);
    llhd_ratio = 0;
    for ks = 1:ns
        k = S(ks);
        
        if Z(k) == ci0
            lamTmp0 = exp(ti0.C(k,:)*ti0.X + ti0.d(k));
        else
            lamTmp0 = exp(tj0.C(k,:)*tj0.X + tj0.d(k));
        end
        lamTmp = exp(tm.C(k,:)*tm.X + tm.d(k));
        
        llhd_ratio = llhd_ratio +...
            sum(log(poisspdf(Y(k,:), lamTmp))) -...
            sum(log(poisspdf(Y(k,:), lamTmp0)));
    end
    p_accept = min(1.0, exp(log_prop_ba-log_prop_ab +...
        log_prior_b-log_prior_a + llhd_ratio));
    
    if rand < p_accept % accept merge
        disp('accept merge')
        for ks = 1:ns
            Z(S(ks)) = cm;
        end
        actList = ordered_remove(ci0, actList, t_fit_tmp);
        actList = ordered_remove(cj0, actList, t_fit_tmp-1);
        actList = ordered_insert(cm, actList, t_fit_tmp-2);
        
        numClus_fit_tmp(cm) = ns;
        numClus_fit_tmp(ci0) = 0;
        numClus_fit_tmp(cj0) = 0;
        t_fit_tmp = t_fit_tmp - 1;
        
        theta(cm) = tm;
    end
end







end