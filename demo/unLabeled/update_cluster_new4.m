function [theta_b, epsilonOut, log_pdf] =...
    update_cluster_new4(Y_tmp,theta_a,theta_b,...
    prior, N, T, p, obsIdx, active, density, OPTDC_tmp)

% for debug
% c=j;
% obsIdx = obsIdx;
% Y_tmp = Y(obsIdx,:);
% theta_a = THETA{g-1}(c);
% theta_b = THETA{g}(c);
% active = true;
% density = false;
% OPTDC_tmp = OPTDC(obsIdx);

N_tmp = size(Y_tmp, 1);
log_pdf = NaN;


% (2) update loading: C
% the transition kernel for NUTS is symmetric --> no need for calculation
epsilonOut = ones(N_tmp, 1)*0.01;

if active
    for i = 1:N_tmp
        X_tmpC = theta_a.X';
        
        lamC = @(c) exp(theta_a.d' + X_tmpC*c);
        
        % use NUTS
        lpdf = @(c) sum(log(poisspdf(Y_tmp(i,:)', lamC(c)))) +...
            log(mvnpdf(c, prior.muC0, prior.SigC0));
        glpdf = @(c) X_tmpC'*(Y_tmp(i,:)' - lamC(c)) - prior.SigC0\(c - prior.muC0);
        
        fg=@(dc_r) deal(lpdf(dc_r'), glpdf(dc_r')'); % log density and gradient
        c0 = theta_a.C(obsIdx(i),:)';
        
        [c_NUTS, ~, diagn]=hmc_nuts(fg, c0',OPTDC_tmp{i});
        epsilonOut(i) = diagn.opt.epsilonbar;
        theta_b.C(obsIdx(i),:) = c_NUTS(end,:);
    end
end



end