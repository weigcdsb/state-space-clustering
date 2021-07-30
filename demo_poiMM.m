%% simulation
k = 3;
T = 1000;
theta1_true = linspace(1, 2, T);
% theta1_true = [repmat(1, 1, round(T/2)) repmat(2, 1, T - round(T/2))]; %
% pass the test!
theta2_true = [repmat(1, 1, round(T/2)) repmat(2, 1, T - round(T/2))];
period = T/2;
theta3_true = 1.5 + sin((2*pi/period)*(1:T));

n1_true = 10;
n2_true = 10;
n3_true = 10;
N = n1_true + n2_true + n3_true;
X = ones(N, T);

lam1_true = exp(theta1_true);
lam2_true = exp(theta2_true);
lam3_true = exp(theta3_true);

rng(2)
Y1 = poissrnd(repmat(lam1_true, n1_true, 1));
Y2 = poissrnd(repmat(lam2_true, n2_true, 1));
Y3 = poissrnd(repmat(lam3_true, n3_true, 1));
Y = [Y1; Y2; Y3];
z_true = repelem(1:3, [n1_true n2_true n3_true]);

% clusterPlot(Y, z_true)

%% begin to do MCMC
% prior
nG = 5;

Z = zeros(nG, N);
RHO = zeros(nG, k);
THETA = zeros(nG, T, k);

Z(1,:) = ones(1, N);
RHO(1,:) = ones(1, k)/k;
% theta0 = ppafilt_poissexp(Y,X,log(mean(mean(Y(:, 1:20)))),1,1,1e-4);

theta0 = log(mean(mean(Y(:, 1:20))));
THETA0 = ppasmoo_poissexp(Y,X,theta0,1,1,1e-4);
THETA(1, :, 1) = THETA0;
THETA(1, :, 2) = THETA0;
THETA(1, :, 3) = THETA0;
delta0 = ones(1, k);

for g = 2:nG
    
    % 1. sample z
    lam1_all_tmp = exp(X .* repmat(THETA(g-1, :, 1), N, 1));
    lam2_all_tmp = exp(X .* repmat(THETA(g-1, :, 2), N, 1));
    lam3_all_tmp = exp(X .* repmat(THETA(g-1, :, 3), N, 1));
    
    logp_tmp = ...
        [log(RHO(g-1,1)) + sum(log(poisspdf(Y, lam1_all_tmp)), 2)...
        log(RHO(g-1,2)) + sum(log(poisspdf(Y, lam2_all_tmp)), 2)...
        log(RHO(g-1,3)) + sum(log(poisspdf(Y, lam3_all_tmp)), 2)];
    
    clus_tmp = mnrnd(ones(N, 1), softmax(logp_tmp')');
    [Z(g,:), ~] = find(clus_tmp');
    
    % 2. sample rho
    nClus = histc(Z(g,:),1:k);
    RHO(g,:) = drchrnd(delta0 + nClus, 1);
    
    % 3. sample theta
    %     [theta1,W1,~] = ppafilt_poissexp(Y(Z(g,:) == 1, :),X(Z(g,:) == 1, :),...
    %         theta0,1,1,1e-4);
    %     [theta2,W2,~] = ppafilt_poissexp(Y(Z(g,:) == 2, :),X(Z(g,:) == 2, :),...
    %         theta0,1,1,1e-4);
    %     [theta3,W3,~] = ppafilt_poissexp(Y(Z(g,:) == 3, :),X(Z(g,:) == 3, :),...
    %         theta0,1,1,1e-4);
    for m = 1:k
        if(sum(Z(g,:) == m) > 0)
            [theta_tmp,W_tmp,~] = ppasmoo_poissexp(Y(Z(g,:) == m, :),X(Z(g,:) == m, :),...
                theta0,1,1,1e-4);
            THETA(g, :, m) = mvnrnd(theta_tmp',W_tmp);
        else
            THETA(g, :, m) = theta0 + detrend(cumsum(randn(round(T),1)*sqrt(1e-4)));
        end
    end
    
    %     subplot(1, 3, 1)
    %     plot(THETA(g, :, 1))
    %     subplot(1, 3, 2)
    %     plot(THETA(g, :, 2))
    %     subplot(1, 3, 3)
    %     plot(THETA(g, :, 3))
    
end

%% plot
for l= 1:nG
    figure(l)
    subplot(1, 2, 1)
    clusterPlot(Y, z_true)
    title('true')
    subplot(1, 2, 2)
    clusterPlot(Y, Z(l,:))
    title("k = " + l + ", nCluster =", length(unique(Z(l,:))))
end






%% animation

for l=1:min(5, nG)
    if l >1
        cla(s1)
        cla(s2)
        cla(s3)
    end
    
    subplot(2, 3, 1)
    clusterPlot(Y, z_true)
    title('true')
    subplot(2, 3, 2)
    clusterPlot(Y, Z(l,:))
    title("k = " + l)
    
    s1 = subplot(2, 3, 4)
    hold on
    plot(THETA(l, :, Z(l,1)), 'r')
    plot(theta1_true, 'k', 'LineWidth', 2)
    hold off
    title('\theta_1')
    
    s2 = subplot(2, 3, 5)
    hold on
    plot(THETA(l, :, Z(l,1 + n1_true)), 'r')
    plot(theta2_true, 'k', 'LineWidth', 2)
    hold off
    title('\theta_2')
    
    s3 = subplot(2, 3, 6)
    hold on
    plot(THETA(l, :, Z(l,1 + n1_true + n2_true)), 'r')
    plot(theta3_true, 'k', 'LineWidth', 2)
    hold off
    title('\theta_3')
    
    pause(0.8)
end




