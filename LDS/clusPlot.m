addpath(genpath('C:\Users\gaw19004\Documents\GitHub\state-space-clustering'));
%%
clear all;close all;clc;
load('C:\Users\gaw19004\Desktop\LDS_backup\new3\cluster\DP_below_alpha5_norm_AQ_diag_rmRedun_1000.mat')


plotFolder = "C:\Users\gaw19004\Desktop\LDS_backup\new3\cluster\plot\DP_below\";
for k = 1:10:500
    clus = figure;
    clusterPlot(Y, Z_fit(:,k)')
%     title("MM, Max Clust = " + kMM + ", k = " + k); % for MM
    title("DP, \alpha = " + alphaDP + ", burin =" + burnIn, ", k = " + k); % for DP
    saveas(clus, plotFolder+k+".png")
end