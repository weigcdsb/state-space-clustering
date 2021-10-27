rng(1)
ng = 1000;
p=1;
p_fit = ones(1,ng);

prior = make_prior(p);

for k = 1:nClus
    THETA{1}(k) = sample_prior2(prior, N, T, p, false);
end

%%
for k = 1:N
    optdc.M=1;
    optdc.Madapt=0;
    OPTDC{k} = optdc;
end

burnIn = 10;
epsilon = 0.01*ones(N,1);

for g = 2:ng
    
    if(g < burnIn);disp("iter " + g + ", changing"); % change epsilon
    elseif(g == burnIn) % tune epsilon
        for(k = 1:N);OPTDC{k}.Madapt=50;end
        disp("iter " + g + ", tuning");
    else;disp("iter " + g + ", tuned");
    end % fix epsilon
    
    THETA{g} = THETA{g-1};
    for j = 1:nClus
        obsIdx = find(Lab == j);
        
        [THETA{g}(j), epsilon(obsIdx), log_pdf] =...
            update_cluster(Y(obsIdx,:),THETA{g-1}(j),THETA{g}(j),...
            prior, N, T, p, obsIdx, true, false, OPTDC(obsIdx));
    end
    
    if(g == burnIn)
        for k = 1:N
            OPTDC{k}.Madapt=0;OPTDC{k}.epsilon = epsilon(k);
        end
    end
    
    
    if(rand < exp(-1 - 10*g/ng))
        
        dropIdx = [];
        for k = 1:p
            zeroNum = 0;
            for j = 1:nClus
                zeroNum = zeroNum + sum(abs(THETA{g}(j).C(Lab == j, k)) < 2);
            end
            
            if (zeroNum > N/2)
                dropIdx = [dropIdx k];
            end
        end
        
        if isempty(dropIdx)
            p = p+1;
            prior = make_prior(p);
            for k = 1:nClus
                THETA_tmp = sample_prior2(prior, N, T, p, false);
                THETA_tmp.mudc(1:(end-1)) = THETA{g}(k).mudc;
%                 THETA_tmp.Sigdc(1:(end-1), 1:(end-1)) = THETA{g}(k).Sigdc;
                THETA_tmp.C(:,1:(end-1)) = THETA{g}(k).C;
                THETA_tmp.A(1:(end-1), 1:(end-1)) = THETA{g}(k).A;
                THETA_tmp.b(1:(end-1)) = THETA{g}(k).b;
                THETA_tmp.Q(1:(end-1), 1:(end-1)) = THETA{g}(k).Q;
                THETA_tmp.Xori(1:(end-1),:) = THETA{g}(k).Xori;
                THETA_tmp.x0(1:(end-1)) = THETA{g}(k).x0;
                THETA_tmp.X(1:(end-1),:) = THETA{g}(k).X;
                
                THETA{g}(k) = THETA_tmp;
            end
        else
            p = p-1;
            prior = make_prior(p);
            for dr = dropIdx(:)'
                for k = 1:nClus
                     THETA{g}(k).mudc(dr+1) = [];
                     THETA{g}(k).Sigdc = THETA{g}(k).Sigdc(1:(end-1),1:(end-1));
                     THETA{g}(k).C(:,dr) = [];
                     THETA{g}(k).A(dr,:) = [];
                     THETA{g}(k).A(:,dr) = [];
                     THETA{g}(k).b(dr) = [];
                     THETA{g}(k).Q(dr,:) = [];
                     THETA{g}(k).Q(:,dr) = [];
                     THETA{g}(k).Xori(dr,:) = [];
                     THETA{g}(k).x0(dr) = [];
                     THETA{g}(k).X(dr,:) = [];
                end
            end
        end
    end
    p_fit(g) = p;
    
    
    
    figure(1)
    subplot(3,1,1)
    plot(THETA{g}(1).X')
    title('fit')
    subplot(3,1,2)
    plot(THETA{g}(2).X')
    subplot(3,1,3)
    plot(THETA{g}(3).X')
    
    figure(2)
    plot(p_fit(1:g))
    
end











