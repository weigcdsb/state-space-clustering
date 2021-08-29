addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
%%
clear all;close all;clc;
load('C:\Users\gaw19004\Desktop\LDS_backup\unspecifiedA_50_DP_above.mat')

plotFolder = "C:\Users\gaw19004\Desktop\LDS_backup\ClusPlots\noA_50_DP_above_1\";
for k = 1:ng
    clus = figure;
    clusterPlot(Y, Z_fit(:,k)')
    title("k = " + k);
    saveas(clus, plotFolder+k+".png")
end