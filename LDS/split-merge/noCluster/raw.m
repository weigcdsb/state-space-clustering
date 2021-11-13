rng(1)
ng = 1000;

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
    
    
    figure(1)
    subplot(3,2,1)
    plot(X(1:p,:)')
    title('true')
    subplot(3,2,2)
    plot(THETA{g}(1).X')
    title('fit')
    subplot(3,2,3)
    plot(X(p+1:2*p,:)')
    subplot(3,2,4)
    plot(THETA{g}(2).X')
    subplot(3,2,5)
    plot(X(2*p+1:3*p,:)')
    subplot(3,2,6)
    plot(THETA{g}(3).X')
    
    figure(2)
    subplot(1,2,1)
    imagesc(exp(C_trans*X + d))
    cLim = caxis;
    title('true')
    colorbar()
    subplot(1,2,2)
    fitMFR = zeros(N, T);
    for k  = 1:3
        fitMFR(Lab == k,:) = exp(THETA{g}(k).d(Lab == k) +...
            THETA{g}(k).C(Lab == k,:)*THETA{g}(k).X);
    end
    imagesc(fitMFR)
    set(gca,'CLim',cLim)
    colorbar()
    title('fit')
end


%%


xMean1 = zeros(p,T);
xMean2 = zeros(p,T);
xMean3 = zeros(p,T);

x50_1 = zeros(p, ng);
x50_2 = zeros(p, ng);
x50_3 = zeros(p, ng);


c = 0;
for g= 1:ng
    
    x50_1(:, g) = THETA{g}(1).X(:, T/2);
    x50_2(:, g) = THETA{g}(2).X(:, T/2);
    x50_3(:, g) = THETA{g}(3).X(:, T/2);
    
    if (g >= 500)
        c= c+1;
        xMean1 = xMean1 + THETA{g}(1).X;
        xMean2 = xMean2 + THETA{g}(2).X;
        xMean3 = xMean3 + THETA{g}(3).X;
    end
end

figure(3)
subplot(3,2,1)
plot(X(1:p,:)')
title('true')
subplot(3,2,2)
plot(xMean1'/c)
title('fit')
subplot(3,2,3)
plot(X(p+1:2*p,:)')
subplot(3,2,4)
plot(xMean2'/c)
subplot(3,2,5)
plot(X(2*p+1:3*p,:)')
subplot(3,2,6)
plot(xMean3'/c)

%% trace plot

figure(4)
hold on
plot(x50_1')
plot(x50_2')
plot(x50_3')
hold off
  






