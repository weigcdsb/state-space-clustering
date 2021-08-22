
plotFolder = "C:\Users\gaw19004\Desktop\LDS_backup\ClusPlots\load_MM\";

for k = 1:60
    clus = figure;
    clusterPlot(Y, Z_fit(:,k)')
    title("k = " + k);
    saveas(clus, plotFolder+"load_MM_"+k+".png")
end