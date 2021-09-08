addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
%%
clear all;close all;clc;
load('C:\Users\gaw19004\Desktop\LDS_backup\new\noA_full_MM_below_4.mat')

plotFolder = "C:\Users\gaw19004\Desktop\LDS_backup\new\plot\noA_full_MM_3\";
for k = 1:ng
    clus = figure;
    clusterPlot(Y, Z_fit(:,k)')
    title("MM, Max Clust = " + kMM + ", k = " + k); % for MM
%     title("DP, \alpha = " + alphaDP + ", k = " + k); % for DP
    saveas(clus, plotFolder+k+".png")
end