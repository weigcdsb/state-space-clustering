addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
% addpath(genpath('D:\github\state-space-clustering'));

plotFolder = 'C:\Users\gaw19004\Documents\GitHub\state-space-clustering\documents\EAC-ISBA\plots\raw';
cd(plotFolder)

%% Simulation 1: labeled
load('C:\Users\gaw19004\Desktop\LDS_backup\new4\ortho_diagAQ_projection.mat')

d_norm_fro = zeros(g,1);
C_norm_fro = zeros(g, 1);
b_norm = zeros(g, 1);
A_norm_fro = zeros(g, 1);
X_norm_fro = zeros(g, 1);
Q_norm_fro = zeros(g, 1);

for k = 1:g
    d_norm_fro(k) = norm(d_fit(:,:,k), 'fro');
    C_norm_fro(k) = norm(C_fit(:,:,k), 'fro');
    b_norm(k) = norm(b_fit(:,k));
    A_norm_fro(k) = norm(A_fit(:,:,k), 'fro');
    X_norm_fro(k) = norm(X_fit(:,:,k), 'fro');
    Q_norm_fro(k) = norm(Q_fit(:,:,k), 'fro');
end

loading = figure;
subplot(2,1,1)% subplot(3,1,1)
plot(d_norm_fro)
title('norm of d')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
subplot(2,1,2)% subplot(3,1,2)
plot(C_norm_fro)
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
xlabel('iteration')
title("Frobenius norm of C, dim: " + N + "\times" + p)

set(loading,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(loading, '1_loading.svg')
saveas(loading, '1_loading.png')



dynamics = figure;
subplot(3,1,1)
plot(b_norm)
set(gca,'FontSize',8, 'LineWidth', 1.5,'TickDir','out')
box off
title('norm of b')
subplot(3,1,2)
plot(A_norm_fro)
set(gca,'FontSize',8, 'LineWidth', 1.5,'TickDir','out')
box off
title('Frobenius norm of A')
subplot(3,1,3)
plot(Q_norm_fro)
xlabel('iteration')
set(gca,'FontSize',8, 'LineWidth', 1.5,'TickDir','out')
box off
title('Frobenius norm of Q')

set(dynamics,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(dynamics, '2_dynamics.svg')
saveas(dynamics, '2_dynamics.png')



gTrunc = 1000;
loading_trunc = figure;
subplot(2,1,1)% subplot(3,1,1)
plot(d_norm_fro(1:gTrunc))
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
subplot(2,1,2)% subplot(3,1,2)
plot(C_norm_fro(1:gTrunc))
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
xlabel('iteration')

set(loading_trunc,'PaperUnits','inches','PaperPosition',[0 0 3 3])
saveas(loading_trunc, '3_loading_trunc.svg')
saveas(loading_trunc, '3_loading_trunc.png')


dynamics_trunc = figure;
subplot(3,1,1)
plot(b_norm(1:gTrunc))
set(gca,'FontSize',8, 'LineWidth', 1.5,'TickDir','out')
box off
subplot(3,1,2)
plot(A_norm_fro(1:gTrunc))
set(gca,'FontSize',8, 'LineWidth', 1.5,'TickDir','out')
box off
subplot(3,1,3)
plot(Q_norm_fro(1:gTrunc))
set(gca,'FontSize',8, 'LineWidth', 1.5,'TickDir','out')
box off
xlabel('iteration')

set(dynamics_trunc,'PaperUnits','inches','PaperPosition',[0 0 3 3])
saveas(dynamics_trunc, '4_dynamics_trunc.svg')
saveas(dynamics_trunc, '4_dynamics_trunc.png')


idx = round(ng/10): ng;

FR = figure;
subplot(1,2,1)
imagesc(exp(C_trans*X + d))
cLim = caxis;
title('true')
colorbar()
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
ylabel('neuron')
subplot(1,2,2)
CX_fit_sum = zeros(N, T);
d_fit_sum = zeros(N,1);
for k = idx
    CX_fit_sum = CX_fit_sum + C_fit(:,:,k)*X_fit(:,:,k);
    d_fit_sum = d_fit_sum + sum(d_fit(:,:,k),2);
end
CX_fit_mean = CX_fit_sum/length(idx);
d_fit_mean = d_fit_sum/length(idx);
imagesc(exp(CX_fit_mean + d_fit_mean))
set(gca,'CLim',cLim)
title('fit')
xlabel('T')
colorbar()
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FR,'PaperUnits','inches','PaperPosition',[0 0 6 3])
saveas(FR, '5_FR.svg')
saveas(FR, '5_FR.png')


FR2 = figure;
subplot(1,2,1)
imagesc(exp(C_trans*X + d))
cLim = caxis;
title('true')
% colorbar()
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
ylabel('neuron')
subplot(1,2,2)
CX_fit_sum = zeros(N, T);
d_fit_sum = zeros(N,1);
for k = idx
    CX_fit_sum = CX_fit_sum + C_fit(:,:,k)*X_fit(:,:,k);
    d_fit_sum = d_fit_sum + sum(d_fit(:,:,k),2);
end
CX_fit_mean = CX_fit_sum/length(idx);
d_fit_mean = d_fit_sum/length(idx);
imagesc(exp(CX_fit_mean + d_fit_mean))
set(gca,'CLim',cLim)
title('fit')
xlabel('T')
% colorbar()
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FR2,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(FR2, '6_FR2.svg')
saveas(FR2, '6_FR2.png')


latent = figure;
subplot(3,2,1)
plot(X(1:p,:)')
title('true')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
subplot(3,2,2)
plot(mean(X_fit(1:p,:,idx), 3)')
title('fit')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
subplot(3,2,3)
plot(X(p+1:2*p,:)')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
subplot(3,2,4)
plot(mean(X_fit(p+1:2*p,:,idx), 3)')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
subplot(3,2,5)
plot(X(2*p+1:3*p,:)')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
subplot(3,2,6)
plot(mean(X_fit(2*p+1:3*p,:,idx), 3)')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
xlabel('T')

set(latent,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(latent, '7_latent.svg')
saveas(latent, '7_latent.png')


%% Simulation 2: unlabeled, n = 10
close all;clc;
load('C:\Users\gaw19004\Desktop\LDS_backup\10272021\sim_3_cluster_10.mat')


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
saveas(sim10, '8_sim10.svg')
saveas(sim10, '8_sim10.png')



sim10_trunc = figure;
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
xlim([0 10])
clusLab = [];
for c = 1:zMax
    clusLab{c} = 'cluster ' + string(c);
end
yticklabels(clusLab)
xlabel('iteration')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(sim10_trunc,'PaperUnits','inches','PaperPosition',[0 0 3 3])
saveas(sim10_trunc, '9_sim10_trunc.svg')
saveas(sim10_trunc, '9_sim10_trunc.png')


%% Simulation 3: unlabeled, n = 30
close all;clc;
load('C:\Users\gaw19004\Desktop\LDS_backup\10272021\sim_3_cluster_30.mat')


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
saveas(sim30, '10_sim30.svg')
saveas(sim30, '10_sim30.png')

FR30 = figure;
imagesc(exp(C_trans*X + d))
cLim = caxis;
title('Spikes')
colorbar()
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
ylabel('neuron')
xlabel('T')

set(FR30,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(FR30, '11_FR30.svg')
saveas(FR30, '11_FR30.png')



%% Application: neural pixel
close all;clc;
load('C:\Users\gaw19004\Desktop\LDS_backup\10272021\4clusters_v2_3000.mat')

idxMax = max(Z_fit((burnIn+1):end));
countMat = zeros(length(unique(Lab)), idxMax, ng-burnIn);
for k = 1:size(countMat, 3)
    for l = 1:idxMax
        idxTmp = find(Z_fit(:, k+burnIn) == l);
        countMat(:,l,k) = histcounts(Lab(idxTmp), 1:5);
    end
end

numEach = histcounts(Lab);

pixel_clus = figure;
meanCount = mean(countMat, 3);
meanCount2 = meanCount(:, sum(meanCount, 1) > 0);
[~, sortIdx] = sort(sum(meanCount2, 1), 'descend');
plotMat = (diag(numEach))\meanCount2(:,sortIdx);
imagesc(plotMat(:,[3 4 1 2 5 6]))
colormap(flipud(hot))
xlabel('cluster-model')
ylabel('cluster-anatomy')
yticks(1:4)
yticklabels(clusIdx)
colorbar()
title('anatomy vs. fitted: proportion')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(pixel_clus,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(pixel_clus, '12_pixel_clus.svg')
saveas(pixel_clus, '12_pixel_clus.png')


pixel_FR = figure;
imagesc(Y)
colorbar()
hold on
numEach = histcounts(Lab);
ytickPos = zeros(1, 4);
for k = 1:4
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
saveas(pixel_FR, '13_pixel_FR.svg')
saveas(pixel_FR, '13_pixel_FR.png')










