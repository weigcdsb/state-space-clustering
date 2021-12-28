
plotFolder = 'C:\Users\gaw19004\Documents\GitHub\state-space-clustering\LDS\split-merge\noCluster\plots';
cd(plotFolder)

%%
dSum1 = zeros(1,T);
dSum2 = zeros(1,T);
dSum3 = zeros(1,T);
xSum1 = zeros(p,T);
xSum2 = zeros(p,T);
xSum3 = zeros(p,T);

dxSum1 = zeros(p+1,T);
dxSum2 = zeros(p+1,T);
dxSum3 = zeros(p+1,T);

c = 0;
for g= 1:ng
    
    if (g >= 500)
        c= c+1;
        dSum1 = dSum1 + THETA{g}(1).d;
        dSum2 = dSum2 + THETA{g}(2).d;
        dSum3 = dSum3 + THETA{g}(3).d;
        
        xSum1 = xSum1 + THETA{g}(1).X;
        xSum2 = xSum2 + THETA{g}(2).X;
        xSum3 = xSum3 + THETA{g}(3).X;
        
        dxSum1 = dxSum1 + [THETA{g}(1).d;THETA{g}(1).X];
        dxSum2 = dxSum2 + [THETA{g}(2).d;THETA{g}(2).X];
        dxSum3 = dxSum3 + [THETA{g}(3).d;THETA{g}(3).X];
    end
end

dxMean1 = dxSum1/c;
dxMean2 = dxSum2/c;
dxMean3 = dxSum3/c;

dxMean1 = dxMean1 - mean(dxMean1, 2);
[dxMean1, ~] = mgson(dxMean1');

dxMean2 = dxMean2 - mean(dxMean2, 2);
[dxMean2, ~] = mgson(dxMean2');

dxMean3 = dxMean3 - mean(dxMean3, 2);
[dxMean3, ~] = mgson(dxMean3');

%% LF
LF = figure;
subplot(3,4,1)
plot(X(id2id(1,2),:)')
y1 = ylim;
title('dX-trans: true')
subplot(3,4,2)
plot(dSum1'/c)
title('d:fit')
subplot(3,4,3)
plot(xSum1'/c)
title('X-raw:fit')
subplot(3,4,4)
plot(dxMean1)
ylim(y1);
title('dX-trans:fit')

subplot(3,4,5)
plot(X(id2id(2,2),:)')
y2 = ylim;
subplot(3,4,6)
plot(dSum2'/c)
subplot(3,4,7)
plot(xSum2'/c)
subplot(3,4,8)
plot(dxMean2)
ylim(y2);

subplot(3,4,9)
plot(X(id2id(3,2),:)')
y3 = ylim;
subplot(3,4,10)
plot(dSum3'/c)
subplot(3,4,11)
plot(xSum3'/c)
subplot(3,4,12)
plot(dxMean3)
ylim(sort(-y3));


set(LF,'PaperUnits','inches','PaperPosition',[0 0 6 3])
saveas(LF, '1_LF.svg')
saveas(LF, '1_LF.png')



%% trace plot
dtrace = zeros(nClus, ng);
Xtrace = zeros(nClus, ng);
Ctrace = zeros(nClus, ng);

btrace = zeros(nClus, ng);
Atrace = zeros(nClus, ng);
Qtrace = zeros(nClus, ng);


for g = 1:ng
    for k = 1:nClus
        dtrace(k,g) = norm(THETA{g}(k).d);
        Xtrace(k,g) = norm(THETA{g}(k).X, 'fro');
        Ctrace(k,g) = norm(THETA{g}(k).C(Lab ==k,:), 'fro');
        btrace(k,g) = norm(THETA{g}(k).b(2:end), 'fro');
        Atrace(k,g) = norm(THETA{g}(k).A(2:end,2:end), 'fro');
        Qtrace(k,g) = norm(THETA{g}(k).Q(2:end,2:end), 'fro');
    end
end

gSub = [1:4000 6000:ng];
ngSub = length(gSub);


trace = figure;
subplot(3,1,1)
plot(dtrace(:,gSub)')
title('Frobenius norm of \mu')
ylim([0 60])
xlim([0 ngSub])
subplot(3,1,2)
plot(Xtrace(:,gSub)')
title('Frobenius norm of X')
ylim([0 20])
xlim([0 ngSub])
subplot(3,1,3)
plot(Ctrace(:,gSub)')
title('Frobenius norm of c')
xlim([0 ngSub])


set(trace,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(trace, '2_trace.svg')
saveas(trace, '2_trace.png')




%% fitted FR
idx = 500:1000;
dSum = zeros(nClus, T);
XSum = zeros(nClus*p, T);
CSum = zeros(N, p);

for g = idx
    for ii = 1:nClus
        dSum(ii,:) = dSum(ii,:) + THETA{g}(ii).d;
        XSum(id2id(ii,p),:) = XSum(id2id(ii,p),:) + THETA{g}(ii).X;
        CSum(Lab == ii, :) = CSum(Lab == ii, :) +...
            THETA{g}(ii).C(Lab == ii, :);
    end
end

dMean = dSum/length(idx);
XMean = XSum/length(idx);
CMean = CSum/length(idx);

fitMFR = zeros(N, T);
for k  = 1:nClus
    N_tmp = sum(Lab == k);
    fitMFR(Lab == k,:) = exp([ones(N_tmp,1) CMean(Lab == k,:)]*...
        [dMean(k,:) ;XMean(id2id(k,p),:)]);
end

subplot

FR = figure;
subplot(1,2,1)
imagesc(exp(C_trans*X + d))
cLim = caxis;
title('true')
ylabel('neuron')
colorbar()
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
subplot(1,2,2)
imagesc(fitMFR)
set(gca,'CLim',cLim)
colorbar()
title('fit')
xlabel('T')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FR,'PaperUnits','inches','PaperPosition',[0 0 6 3])
saveas(FR, '2_FR.svg')
saveas(FR, '2_FR.png')

%%
%% cluster plot...

sim10 = figure;
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
clusLab = [];
for c = 1:zMax
    clusLab{c} = 'cluster ' + string(c);
end
yticklabels(clusLab)
title('cluster trace for each neuron')
xlabel('iteration')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(sim10,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(sim10, '3_sim10.svg')
saveas(sim10, '3_sim10.png')


sim30 = figure;
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
clusLab = [];
for c = 1:zMax
    clusLab{c} = 'cluster ' + string(c);
end
yticklabels(clusLab)
title('cluster trace for each neuron')
xlabel('iteration')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(sim30,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(sim30, '4_sim30.svg')
saveas(sim30, '4_sim30.png')

clusLast = figure;
clusterPlot(Y, Z_fit(:,ng)')
set(clusLast,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(clusLast, '5_sim30Last.svg')
saveas(clusLast, '5_sim30Last.png')




FR20 = figure;
% imagesc(exp(C_trans*X + d))
imagesc(Y)
cLim = caxis;
title('Spikes')
colorbar()
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
ylabel('neuron')
xlabel('T')

set(FR20,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(FR20, '6_FR20.svg')
saveas(FR20, '6_FR20.png')


sim20 = figure;
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

set(sim20,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(sim20, '7_sim20.svg')
saveas(sim20, '7_sim20.png')

hist20 = figure;
histogram(t_fit(idx),'Normalization','probability')

set(hist20,'PaperUnits','inches','PaperPosition',[0 0 3 3])
saveas(hist20, '8_FR20.svg')
saveas(hist20, '8_FR20.png')


%% application...







