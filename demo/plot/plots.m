addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));

%% true
plotFolder = 'C:\Users\gaw19004\Documents\GitHub\state-space-clustering\demo\globalNonlinear\plots';
cd(plotFolder)

dxSum3 = zeros(p+1,T);
dxSum5 = zeros(p+1,T);

c = 0;
for g = round(ng/2):ng
    c= c+1;
    dxSum3 = dxSum3 + [THETA{g}(3).d;THETA{g}(3).X];
    dxSum5 = dxSum5 + [THETA{g}(5).d;THETA{g}(5).X];
end

dxMean3 = dxSum3/c;
dxMean3 = dxMean3 - mean(dxMean3, 2);
[dxMean3, ~] = mgson(dxMean3');

dxMean5 = dxSum5/c;
dxMean5 = dxMean5 - mean(dxMean5, 2);
[dxMean5, ~] = mgson(dxMean5');

hold on
plot(-X(5,:), 'k', 'LineWidth', 1)
plot(dxMean3(:,1), 'r', 'LineWidth', 1)
plot(dxMean5(:,1), 'b', 'LineWidth', 1)
hold off





%% no cluster
clear all;close all;clc;
% plotFolder = 'C:\Users\gaw19004\Documents\GitHub\state-space-clustering\plots\noCluster';
plotFolder = 'C:\Users\gaw19004\Documents\GitHub\state-space-clustering\demo\globalNonlinear\plots\noCluster';
cd(plotFolder)

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

% LF

LF = figure;
subplot(3,2,1)
hold on
plot(X(1,:), 'k', 'LineWidth', 1)
plot(-dxMean1(:,1), 'r', 'LineWidth', 1)
hold off
ylabel("Clus. 1")
title("L1")
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out','ytick',[])
box off


subplot(3,2,2)
hold on
plot(X(2,:), 'k', 'LineWidth', 1)
plot(dxMean1(:,2), 'b', 'LineWidth', 1)
hold off
title('L2')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out','ytick',[])
box off

subplot(3,2,3)
hold on
plot(X(3,:), 'k', 'LineWidth', 1)
plot(-dxMean2(:,1), 'r', 'LineWidth', 1)
hold off
ylabel("Clus. 2")
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out','ytick',[])
box off

subplot(3,2,4)
hold on
plot(X(4,:), 'k', 'LineWidth', 1)
plot(-dxMean2(:,2), 'b', 'LineWidth', 1)
hold off
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out','ytick',[])
box off

subplot(3,2,5)
hold on
plot(X(5,:), 'k', 'LineWidth', 1)
plot(-dxMean3(:,1), 'r', 'LineWidth', 1)
hold off
ylabel("Clus. 3")
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out','ytick',[])
box off

subplot(3,2,6)
hold on
plot(X(6,:), 'k', 'LineWidth', 1)
plot(-dxMean3(:,2), 'b', 'LineWidth', 1)
hold off
xlabel('T')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out','ytick',[])
box off


set(LF,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(LF, '1_LF.svg')
saveas(LF, '1_LF.png')


fitMFRTrace = zeros(N,T,ng);
tracellhd = zeros(ng,1);

for g = 1:ng
    for k  = 1:N
        fitMFRTrace(k,:, g) = exp([1 THETA{g}(Lab(k)).C(k,:)]*...
            [THETA{g}(Lab(k)).d ;THETA{g}(Lab(k)).X]);
    end
    tracellhd(g) =...
        nansum(log(poisspdf(Y,fitMFRTrace(:,:,g))), 'all')/nansum(Y, 'all');
end


llhdspk1 = figure;
plot(tracellhd)
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
xlabel('iteration')
ylabel('llhd/spk')
box off

set(llhdspk1,'PaperUnits','inches','PaperPosition',[0 0 8 3])
saveas(llhdspk1, '2_llhd1.svg')
saveas(llhdspk1, '2_llhd1.png')


llhdspk2 = figure;
plot(tracellhd(1:1000))
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
xlabel('iteration')
ylabel('llhd/spk')
box off

set(llhdspk2,'PaperUnits','inches','PaperPosition',[0 0 6 3])
saveas(llhdspk2, '3_llhd2.svg')
saveas(llhdspk2, '3_llhd2.png')


idx = round(ng/2):ng;
FR = figure;
subplot(1,2,1)
imagesc(exp(C_trans*X + d))
hold on
yPos = zeros(nClus,1);
yticLab = {};
for cc = 1:nClus
    yline(n*cc+0.5, 'y--', 'LineWidth', 2);
    yPos(cc) = (cc-0.5)*n;
    yticLab{cc} = "cluster" + cc;
end
yticks(yPos)
yticklabels(yticLab)
hold off
cLim = caxis;
title('true')
% ylabel('neuron')
colorbar()
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
subplot(1,2,2)
imagesc(mean(fitMFRTrace(:,:,idx), 3))
hold on
for cc = 1:nClus
    yline(n*cc+0.5, 'y--', 'LineWidth', 2);
end
yticks(yPos)
yticklabels(yticLab)
hold off
set(gca,'CLim',cLim)
colorbar()
title('fit')
xlabel('T')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FR,'PaperUnits','inches','PaperPosition',[0 0 6 3])
saveas(FR, '4_FR.svg')
saveas(FR, '4_FR.png')

%% speckle cross-validation...
clear all; close all;clc;
% plotFolder = 'C:\Users\gaw19004\Documents\GitHub\state-space-clustering\plots\speckle';
plotFolder = 'C:\Users\gaw19004\Documents\GitHub\state-space-clustering\demo\globalNonlinear\plots\noCluster_compare';
cd(plotFolder)

% bp = figure;
% boxplot([llhd_spk_test_3clus(:,1) - llhd_spk_test_1clus(:,1);...
%     llhd_spk_test_3clus(:,2) - llhd_spk_test_1clus(:,2);...
%     llhd_spk_test_3clus(:,1) - llhd_spk_test_1clus_tune(:,1);...
%     llhd_spk_test_3clus(:,2) - llhd_spk_test_1clus_tune(:,2)],...
%     repelem(["3/4T: 3c-1c_p5" "1/2T: 3c-1c_p5"...
%     "3/4T: 3c-1c_pT" "1/2T: 3c-1c_pT"], nRep))
% set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
% box off
% 
% set(bp,'PaperUnits','inches','PaperPosition',[0 0 6 3])
% saveas(bp, '1_bp.svg')
% saveas(bp, '1_bp.png')
% 
% 
% llhdAll = figure;
% llhdTest1 = [llhd_spk_test_3clus(:,1) ...
%     llhd_spk_test_1clus(:,1) ...
%     llhd_spk_test_1clus_tune(:,1)];
% 
% llhdTest2 = [llhd_spk_test_3clus(:,2) ...
%     llhd_spk_test_1clus(:,2) ...
%     llhd_spk_test_1clus_tune(:,2)];
% 
% 
% subplot(2,1,1)
% hold on
% plot(llhdTest1','Color', [0.5, 0.5, 0.5, 0.3])
% q1 = quantile(llhdTest1,0.25);
% medVal = median(llhdTest1);
% q3 = quantile(llhdTest1,0.75);
% meanVal = mean(llhdTest1);
% plot(medVal,'r', 'LineWidth',2)
% plot(meanVal, 'b', 'LineWidth',2)
% plot(q1, 'r--', 'LineWidth',2)
% plot(q3, 'r--', 'LineWidth',2)
% for t = 1:numel(medVal)
%   text(t-0.25,medVal(t)+0.008,num2str(round(medVal(t),3)),'Color','red','FontSize',15)
%   text(t-0.25,medVal(t)-0.008,num2str(round(medVal(t),3)),'Color','blue','FontSize',15)
% end
% hold off
% xlim([0.5 3.5])
% xticks(1:3)
% xticklabels({'3cluster - p1','1cluster - p5','1cluster - pTune'})
% title('3/4-train')
% set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
% box off
% 
% subplot(2,1,2)
% hold on
% plot(llhdTest2','Color', [0.5, 0.5, 0.5, 0.3])
% q1 = quantile(llhdTest2,0.25);
% medVal = median(llhdTest2);
% q3 = quantile(llhdTest2,0.75);
% meanVal = mean(llhdTest2);
% plot(medVal,'r', 'LineWidth',2)
% plot(meanVal, 'b', 'LineWidth',2)
% plot(q1, 'r--', 'LineWidth',2)
% plot(q3, 'r--', 'LineWidth',2)
% for t = 1:numel(medVal)
%   text(t-0.25,medVal(t)+0.008,num2str(round(medVal(t),3)),'Color','red','FontSize',15)
%   text(t-0.25,medVal(t)-0.008,num2str(round(medVal(t),3)),'Color','blue','FontSize',15)
% end
% hold off
% xlim([0.5 3.5])
% xticks(1:3)
% xticklabels({'3cluster - p1','1cluster - p5','1cluster - pTune'})
% title('1/2-train')
% set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
% box off
% 
% set(llhdAll,'PaperUnits','inches','PaperPosition',[0 0 6 3])
% saveas(llhdAll, '2_llhdAll.svg')
% saveas(llhdAll, '2_llhdAll.png')


bp_data = [llhd_spk_test_3clus(:,1);...
    llhd_spk_test_1clus_tune(:,1);...
    llhd_spk_test_3clus(:,2);...
    llhd_spk_test_1clus_tune(:,2)];
bp_lab = repelem(["3/4-10C."...
    "3/4-1C."...
    "1/2-10C."...
    "1/2-1C."], nRep);

c = [repelem([0.8500 0.9290 0 0]', nRep),...
repelem([0.3250 0.6940 0.4470 0.7450]', nRep),...
repelem([0.0980 0.1250 0.7410 0.9330]', nRep)];


bp1 = figure;
hold on
s = scatter(repelem(1:4, nRep) + randn(1, 4*nRep)*0.1, bp_data,...
    25, c, 'fill');
boxplot(bp_data,bp_lab, 'Colors', 'k','BoxStyle', 'filled')
alpha(s,.6)
hold off
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(bp1,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(bp1, '1_bpAll.svg')
saveas(bp1, '1_bpAll.png')


bp_data = [llhd_spk_test_3clus(:,1)-...
    llhd_spk_test_1clus_tune(:,1);...
    llhd_spk_test_3clus(:,2)-...
    llhd_spk_test_1clus_tune(:,2)];
bp_lab = repelem(["3/4-(10C.-1C.)"...
    "1/2-(10C.-1C.)"], nRep);
c = [repelem([0.8500  0 ]', nRep),...
repelem([0.3250 0.4470]', nRep),...
repelem([0.0980 0.7410]', nRep)];

bp2 = figure;
hold on
s = scatter(repelem(1:2, nRep) + randn(1, 2*nRep)*0.1, bp_data,...
    25, c, 'fill');
boxplot(bp_data,bp_lab, 'Colors', 'k','BoxStyle', 'filled')
alpha(s,.6)
hold off
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(bp2,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(bp2, '2_bpDiff.svg')
saveas(bp2, '2_bpDiff.png')





%% cluster plot...
clear all; close all;clc;
% plotFolder = 'C:\Users\gaw19004\Documents\GitHub\state-space-clustering\plots\cluster';
plotFolder = 'C:\Users\gaw19004\Documents\GitHub\state-space-clustering\demo\globalNonlinear\plots\cluster';
cd(plotFolder)


fitMFRTrace_all = fitMFRTrace;
llhd_trace_all = llhd_trace;
Z_fit_all = Z_fit;
t_fit_all = t_fit;

fitMFRTrace_half = fitMFRTrace;
llhd_trace_half  = llhd_spk_train;

llhdTrace = figure;
hold on
plot([nan;llhd_trace_all(2:end)], 'r', 'LineWidth', 2)
plot([nan;llhd_trace_half(2:end)], 'b', 'LineWidth', 2)
hold off
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
xlabel('iteration')
ylabel('Training llhd/spk')
box off

set(llhdTrace,'PaperUnits','inches','PaperPosition',[0 0 10 3])
saveas(llhdTrace, '1_llhdTrace.svg')
saveas(llhdTrace, '1_llhdTrace.png')

idx = round(ng/2):ng;
FR20 = figure;
subplot(1,2,1)
imagesc(exp(C_trans*X + d))
nClus = 10;
hold on
yPos = zeros(nClus,1);
yticLab = {};
for cc = 1:nClus
    yline(n*cc+0.5, 'y--', 'LineWidth', 2);
    yPos(cc) = (cc-0.5)*n;
    yticLab{cc} = "cluster" + cc;
end
yticks(yPos)
% yticklabels(yticLab)
hold off
cLim = caxis;
title('true')
colorbar()
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

subplot(1,2,2)
imagesc(mean(fitMFRTrace_all(:,:,idx), 3))
hold on
for cc = 1:nClus
    yline(n*cc+0.5, 'y--', 'LineWidth', 2);
end
yticks(yPos)
% yticklabels(yticLab)
hold off
set(gca,'CLim',cLim)
colorbar()
title('fit')
xlabel('T')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FR20,'PaperUnits','inches','PaperPosition',[0 0 6 3])
saveas(FR20, '2_FR20_all.svg')
saveas(FR20, '2_FR20_all.png')


FR20 = figure;
subplot(1,2,1)
imagesc(exp(C_trans*X + d))
hold on
yPos = zeros(nClus,1);
yticLab = {};
for cc = 1:nClus
    yline(n*cc+0.5, 'y--', 'LineWidth', 2);
    yPos(cc) = (cc-0.5)*n;
    yticLab{cc} = "cluster" + cc;
end
yticks(yPos)
% yticklabels(yticLab)
hold off
cLim = caxis;
title('true')
colorbar()
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

subplot(1,2,2)
imagesc(mean(fitMFRTrace_half(:,:,idx), 3))
hold on
for cc = 1:nClus
    yline(n*cc+0.5, 'y--', 'LineWidth', 2);
end
yticks(yPos)
% yticklabels(yticLab)
hold off
set(gca,'CLim',cLim)
colorbar()
title('fit')
xlabel('T')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FR20,'PaperUnits','inches','PaperPosition',[0 0 6 3])
saveas(FR20, '3_FR20_half.svg')
saveas(FR20, '3_FR20_half.png')


sim20 = figure;
simMat = zeros(N,N);
for g = idx
    for k = 1:size(simMat, 1)
        simMat(k,:) = simMat(k,:) + (Z_fit_all(k, g) == Z_fit_all(:, g))';
    end
end
imagesc(simMat/length(idx))
hold on
for cc = 1:nClus
    yline(n*cc+0.5, 'k--', 'LineWidth', 1);
    xline(n*cc+0.5, 'k--', 'LineWidth', 1);
end
hold off
colormap(flipud(hot))
colorbar()
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(sim20,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(sim20, '4_sim20_all.svg')
saveas(sim20, '4_sim20_all.png')


sim20 = figure;
simMat = zeros(N,N);
for g = idx
    for k = 1:size(simMat, 1)
        simMat(k,:) = simMat(k,:) + (Z_fit(k, g) == Z_fit(:, g))';
    end
end
imagesc(simMat/length(idx))
hold on
for cc = 1:nClus
    yline(n*cc+0.5, 'k--', 'LineWidth', 1);
    xline(n*cc+0.5, 'k--', 'LineWidth', 1);
end
hold off
colormap(CustomColormap)
colorbar()
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(sim20,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(sim20, '5_sim20_half.svg')
saveas(sim20, '5_sim20_half.png')


numTrace = figure;
hold on
plot(t_fit_all, 'r', 'LineWidth', 2)
plot(t_fit, 'b', 'LineWidth', 2)
hold off
ylabel('number of cluster')
xlabel('iteration')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(numTrace,'PaperUnits','inches','PaperPosition',[0 0 10 3])
saveas(numTrace, '6_numTrace.svg')
saveas(numTrace, '6_numTrace.png')


llhd_test_trace = figure;
hold on
plot([nan;llhd_spk_test_true(2:end)], 'LineWidth', 1)
plot([nan;llhd_spk_test(2:end)], 'LineWidth', 1)
plot([nan;llhd_spk_test_sing2(2:end)], 'LineWidth', 1)
hold off
xlabel('iteration')
ylabel('held-out llhd/spk')
% xlim([50 ng])
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(llhd_test_trace,'PaperUnits','inches','PaperPosition',[0 0 3 3])
saveas(llhd_test_trace, '7_llhd_test_trace.svg')
saveas(llhd_test_trace, '7_llhd_test_trace.png')


llhdTest_hist = figure;
hold on
histogram(llhd_spk_test_true(idx))
histogram(llhd_spk_test(idx))
histogram(llhd_spk_test_sing2(idx))
hold off
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
ylabel("Frequency")
xlabel("held-out llhd/spk")
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(llhdTest_hist,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(llhdTest_hist, '8_llhd_test_hist.svg')
saveas(llhdTest_hist, '8_llhd_test_hist.png')

%% application...
clear all;close all;clc;
% plotFolder = 'C:\Users\gaw19004\Documents\GitHub\state-space-clustering\plots\pixel\spontaneous1';
plotFolder = 'C:\Users\gaw19004\Documents\GitHub\state-space-clustering\plots\pixel\spontaneous2';
% plotFolder = 'C:\Users\gaw19004\Documents\GitHub\state-space-clustering\plots\pixel\natural1';
% plotFolder = 'C:\Users\gaw19004\Documents\GitHub\state-space-clustering\plots\pixel\drift1';
% plotFolder = 'C:\Users\gaw19004\Documents\GitHub\state-space-clustering\plots\pixel\spontaneous2_v2';

cd(plotFolder)

pixel_FR = figure;
imagesc(Y)
colorbar()
hold on
numEach = histcounts(Lab);
nRegion = length(unique(Lab));
ytickPos = zeros(1, nRegion);
for k = 1:nRegion
    yline(sum(numEach(1:k)), 'y--', 'LineWidth', 4);
    ytickPos(k) = sum(numEach(1:(k-1))) + numEach(k)/2;
end
yticks(ytickPos)
yticklabels(clusIdx)
hold off
title('spking counts')
xlabel('T')
set(gca,'FontSize',9, 'LineWidth', 1.5,'TickDir','out')
box off

set(pixel_FR,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(pixel_FR, '1_pixel_FR.svg')
saveas(pixel_FR, '1_pixel_FR.png')


numTrace = figure;
plot(t_fit)
ylabel('number of cluster')
xlabel('iteration')
set(gca,'FontSize',9, 'LineWidth', 1.5,'TickDir','out')
box off

set(numTrace,'PaperUnits','inches','PaperPosition',[0 0 10 3])
saveas(numTrace, '2_numTrace.svg')
saveas(numTrace, '2_numTrace.png')

idx = round(ng/2):ng;
histPixel = figure;
histogram(t_fit(idx),'Normalization','probability')
set(gca,'FontSize',9, 'LineWidth', 1.5,'TickDir','out')
box off
xlabel("Number of cluster")
ylabel("Frequency")

set(histPixel,'PaperUnits','inches','PaperPosition',[0 0 2.5 3])
saveas(histPixel, '3_hist.svg')
saveas(histPixel, '3_hist.png')


simPixel = figure;
simMat = zeros(N,N);
for g = idx
    for k = 1:size(simMat, 1)
        simMat(k,:) = simMat(k,:) + (Z_fit(k, g) == Z_fit(:, g))';
    end
end

sortId = zeros(N,1);
for k = 1:length(unique(Lab))
    [~, idTmp] = sort(mode(Z_fit(Lab == k,idx), 2));
    sortId(Lab == k) = sum(Lab < k) + idTmp;
end

imagesc(simMat(sortId,sortId)/length(idx))
hold on
tickPos = zeros(1, nRegion);
for k = 1:nRegion
    yline(sum(numEach(1:k) + 0.5), 'b--', 'LineWidth', 2);
    xline(sum(numEach(1:k) + 0.5), 'b--', 'LineWidth', 2);
    tickPos(k) = sum(numEach(1:(k-1))) + numEach(k)/2;
end
yticks(tickPos)
yticklabels(clusIdx)
xticks(tickPos)
xticklabels(clusIdx)
hold off
colormap(flipud(hot))
colorbar()
set(gca,'FontSize',9, 'LineWidth', 1.5,'TickDir','out')
box off

set(simPixel,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(simPixel, '4_simPixel.svg')
saveas(simPixel, '4_simPixel.png')


fitFR = figure;
subplot(1,2,1)
imagesc(Y)
for k = 1:nRegion
    yline(sum(numEach(1:k)), 'y--', 'LineWidth', 4);
end
yticks(ytickPos)
yticklabels(clusIdx)
colorbar()
title('Spike Counts')
set(gca,'FontSize',9, 'LineWidth', 1.5,'TickDir','out')
box off
subplot(1,2,2)
imagesc(mean(fitMFRTrace(:,:,idx),3))
for k = 1:nRegion
    yline(sum(numEach(1:k)), 'y--', 'LineWidth', 4);
end
yticks(ytickPos)
yticklabels(clusIdx)
colorbar()
title('fit-FR')
xlabel("T")
set(gca,'FontSize',9, 'LineWidth', 1.5,'TickDir','out')
box off

set(fitFR,'PaperUnits','inches','PaperPosition',[0 0 6 3])
saveas(fitFR, '5_fitFR.svg')
saveas(fitFR, '5_fitFR.png')


tracellhd = zeros(ng,1);
for g = 1:ng
    tracellhd(g) =...
        nansum(log(poisspdf(Y,fitMFRTrace(:,:,g))), 'all')/nansum(Y, 'all');
end

traceLLhd_all = figure;
plot(tracellhd)
ylabel('llhd/spk')
xlabel('iteration')
set(gca,'FontSize',9, 'LineWidth', 1.5,'TickDir','out')
box off

set(traceLLhd_all,'PaperUnits','inches','PaperPosition',[0 0 10 3])
saveas(traceLLhd_all, '6_llhdTrace1.svg')
saveas(traceLLhd_all, '6_llhdTrace1.png')


traceLLhd_part = figure;
plot(tracellhd(1:100))
ylabel('llhd/spk')
xlabel('iteration')
set(gca,'FontSize',9, 'LineWidth', 1.5,'TickDir','out')
box off

set(traceLLhd_part,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(traceLLhd_part, '7_llhdTrace2.svg')
saveas(traceLLhd_part, '7_llhdTrace2.png')

%% compare...
clear all;close all;clc;
plotFolder = 'C:\Users\gaw19004\Documents\GitHub\state-space-clustering\plots\pixel';
cd(plotFolder)

comp = figure;
hold on
plot([nan;llhd_spk_test2(2:end)], 'LineWidth', 2)
plot([nan;llhd_spk_test_sing2(2:end)], 'LineWidth', 2)
plot([nan llhd_spk_test_sing_clus(2:end)], 'LineWidth', 2)
hold off
ylabel("held-out llhd/spk")
xlabel("iteration")
set(gca,'FontSize',9, 'LineWidth', 1.5,'TickDir','out')
box off

set(comp,'PaperUnits','inches','PaperPosition',[0 0 5 2])
saveas(comp, '1_spontaneous_comp.svg')
saveas(comp, '1_spontaneous_comp.png')




