
k=1;
% xrange = min(log(Y(k,:) + 1e-2)):0.01:max(log(Y(k,:)));
% xrange = (log(mean(Y(k,:)))-3):0.01:(log(mean(Y(k,:)))+3);

k = 30;

xrange = (log(k)-.25):0.01:(log(k)+.25);

cehbTmp = compute_cheb([min(xrange), max(xrange)]);
% cehbTmp = compute_cheb(log(mean(Y(k,:))) + [-2 2]);
% cehbTmp = compute_cheb(log(mean(Y(k,:))) + [-3 3]);

fApp = @(x) cehbTmp(3)*x.^2 + cehbTmp(2).*x + cehbTmp(1);


hold on
plot(xrange, exp(xrange))
plot(xrange, fApp(xrange))
hold off


yApp = [];
yTrue = [];
kEnd = 50;
xAll = [];

for k = 0:kEnd
    
    if k ==0
        cehbTmp = compute_cheb([log(0.1) log(1)]);
        xrange = log(0.1):0.01:log(1);
    else
        cehbTmp = compute_cheb([log(k) log(k+1)]);
        xrange = log(k):0.01:log(k+1);
    end
    
    fApp = @(x) cehbTmp(3)*x.^2 + cehbTmp(2).*x + cehbTmp(1);
    xAll = [xAll xrange];
    yTrue = [yTrue exp(xrange)];
    yApp = [yApp fApp(xrange)];
    
end

hold on
plot(xAll, yTrue)
plot(xAll, yApp)
hold off







