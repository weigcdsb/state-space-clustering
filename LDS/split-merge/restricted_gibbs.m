function [zsb,tib,tjb,cib,cjb,log_p, ni, nj, iIdx, jIdx] =...
    restricted_gibbs(zsa,zsb,tia,tib,tja,tjb,cia,cib,cja,cjb,...
    iIdx,jIdx, ni,nj,ism,jsm,S,ns,Y,b,prior, N, T, p, active,OPTDC)

% for debug
% zsa = zs;
% zsb = zs;
% tia = ti;
% tib = ti;
% tja = tj;
% tjb = tj;
% cia = ci;
% cib = ci;
% cja = cj;
% cjb = cj;
% active = true;

log_p = 0;
for ks = 1:ns
    k = S(ks);
    if k~= ism && k~=jsm
        if zsa(k) == cia
            ni = ni-1;
        else
            nj = nj-1;
        end
        
        lami = exp(tia.C(k,:)*tia.X + tia.d(k));
        Li = sum(log(poisspdf(Y(k,:), lami)));
        lamj = exp(tja.C(k,:)*tja.X + tja.d(k));
        Lj = sum(log(poisspdf(Y(k,:), lamj)));
        Pi = exp(log(ni+b)+Li - logsumexp(log(ni+b)+Li,log(nj+b)+Lj));
        
        
        if active
            if rand < Pi
                if zsa(k) == cja
                    jIdx = setdiff(jIdx, k);
                    iIdx = [iIdx; k];
                end
                zsb(k) = cib;
            else
                if zsa(k) == cia
                    iIdx = setdiff(iIdx, k);
                    jIdx = [jIdx; k];
                end
                zsb(k) = cjb;
            end
        end
        if zsb(k) == cib
            ni = ni+1;
            log_p = log_p + log(Pi);
        else
            nj = nj + 1;
            log_p = log_p + log(1-Pi);
        end
    end
end

[tib, ~, log_pi] =...
    update_cluster(Y(iIdx,:),tia,tib,...
    prior, N, T, p, iIdx, active, true, OPTDC(iIdx));
[tjb, ~, log_pj] =...
    update_cluster(Y(jIdx,:),tja,tjb,...
    prior, N, T, p, jIdx, active, true, OPTDC(jIdx));
log_p = log_p + log_pi + log_pj;


end