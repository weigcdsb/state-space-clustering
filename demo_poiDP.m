%% simulation
k = 3;
T = 1000;
theta1_true = linspace(2, 1, T);
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

rng(3)
Y1 = poissrnd(repmat(lam1_true, n1_true, 1));
Y2 = poissrnd(repmat(lam2_true, n2_true, 1));
Y3 = poissrnd(repmat(lam3_true, n3_true, 1));
Y = [Y1; Y2; Y3];
z_true = repelem(1:3, [n1_true n2_true n3_true]);

% clusterPlot(Y, z_true)
%%
alphaDP = 1;
nG = 50;
theta0 = log(mean(mean(Y(:, 1:20))));
THETA0 = ppasmoo_poissexp(Y,X,theta0,1,1,1e-4);

Z = zeros(nG, N);
THETA{1} = THETA0; 

% start from single cluster
% Z(1,:) = ones(1, N);

% start from each full clusters
Z(1,:) = 1:N;

for g = 2:nG
    
    % 1. update eta: lower
    z_star = max(Z(g-1,:));
    eta_tmp = zeros(1, z_star);
    for m = 1:z_star
        eta_tmp(m) = betarnd(1 + sum(Z(g-1,:) == m),...
            N - sum(Z(g-1,:) <= m)+ alphaDP);
    end
    
    rho_tmp = eta2rho(eta_tmp);
    z_rho_tab = table((1:z_star)', rho_tmp','VariableNames',{'z', 'rho'});
    
    % 2. update u
    z_tab = table(Z(g-1,:)','VariableNames',{'z'});
    u_tmp = rand(N, 1).*...
        join(z_tab, z_rho_tab).rho;
    
    % 3. update eta: upper
    eta_tmp2 = etaExt(eta_tmp, u_tmp, alphaDP);
    s_star = length(eta_tmp2);
    
    % 4. update theta
    THETA_tmp = zeros(s_star, T);
    for k = 1:s_star
        if sum(Z(g-1,:) == k) > 0
            [theta_tmp,W_tmp,~] = ppasmoo_poissexp(Y(Z(g-1,:) == k, :),X(Z(g-1,:) == k, :),...
                theta0,1,1,1e-4);
            THETA_tmp(k, :) = mvnrnd(theta_tmp',W_tmp);
        else
            THETA_tmp(k, :) = theta0 + detrend(cumsum(randn(round(T),1)*sqrt(1e-4)));
        end
        
    end
    THETA{g} = THETA_tmp;
    
    % 5. update z
    LAM_tmp = zeros(N, T, s_star);
    LLHD = zeros(N, s_star);
    for k=1:s_star
        LAM_tmp(:,:,k) = exp(X .* repmat(THETA_tmp(k,:), N, 1));
        LLHD(:, k) = sum(log(poisspdf(Y, LAM_tmp(:,:,k))), 2);
    end
    
    rho_tmp2 = eta2rho(eta_tmp2);
    LLHD2 = ones(N, s_star)*-Inf;
    LLHD2(u_tmp < rho_tmp2) = LLHD(u_tmp < rho_tmp2);
    
    clus_tmp = mnrnd(ones(N, 1), softmax(LLHD2')');
    [Z(g,:), ~] = find(clus_tmp');
    

end

%% plot
for l= [1 linspace(10, nG, nG/10)]
    clusTrace = figure;
    subplot(1, 2, 1)
    clusterPlot(Y, z_true)
    title('true')
    subplot(1, 2, 2)
    clusterPlot(Y, Z(l,:))
    title("k = " + l + ", nCluster =", length(unique(Z(l,:))))
    saveas(clusTrace, l+".png")
end


for l= [2 linspace(10, nG, nG/10)]
    figure(l + nG)
    THETA_plot = THETA{l}(unique(Z(l,:)), :);
    nClus = size(THETA_plot, 1);
    for p=1:nClus
       subplot(ceil(sqrt(nClus)), ceil(nClus/sqrt(nClus)), p)
       plot(THETA_plot(p,:))
    end
end


subplot(1,3,1)
hold on
plot(theta1_true, 'k', 'LineWidth', 2)
plot(THETA{nG}(Z(nG, 1), :))
hold off
title('\theta_1')
subplot(1,3,2)
hold on
plot(theta2_true, 'k', 'LineWidth', 2)
plot(THETA{nG}(Z(nG, 1+n1_true), :))
hold off
title('\theta_2')
subplot(1,3,3)
hold on
plot(theta3_true, 'k', 'LineWidth', 2)
plot(THETA{nG}(Z(nG, 1+n1_true+n2_true), :))
hold off
title('\theta_3')

