function theta =...
    sample_prior_bt(prior, N, T, p, sampleDynamics, maxCount)

flag = 1;
while flag
    % lodaing: C
    theta.C = randn(N, p);
    theta.muC = theta.C;
    
    % linear dynamics: b, A & Q
    theta.A = eye(p+1);
    theta.b = zeros(p+1,T-1);
    theta.Q = eye(p+1)*prior.sig20;
    
    % mean & latent: d, X
    dX = ones(1+p,T)*Inf;
    dX(:,1) = mvnrnd(prior.theta0, prior.Q0);
    
    if sampleDynamics
        
        for k = 1:(p+1)
            
            theta.A(k,k) = normrnd(prior.A0, prior.sig2A0);
            theta.Q(k,k) = 1/gamrnd(prior.nu0/2, 2/(prior.nu0*prior.sig20));
            for t = 1:(T-1)
                theta.b(k, t) = normrnd(prior.b0,prior.sig2b0);
            end
        end
    end
    
    for t= 2:T
        dX(:, t) = mvnrnd(theta.b(:,t-1) + theta.A*dX(:, t-1), theta.Q);
    end
    
    theta.d = dX(1,:);
    theta.X = dX(2:end,:);
    
    % test
%     theta.X(abs(theta.X) > 1) = sign(theta.X(abs(theta.X) > 1));
    
    
%     dX = zeros(p+1, T);
%     for i=1:(p+1)
%         k = ceil(rand()*20)+10;
%         if i == 1
%             pt = randn(k,1);
%         else
%             pt = randn(k,1)*.25;
%         end
%         dX(i,:) = interp1(linspace(0,1,k),pt,linspace(0,1,T),'spline');
%     end
% 
%     theta.d = dX(1,:);
%     theta.X = dX(2:end,:);
    
    
    
    
    lamTmp = exp([ones(N,1) theta.C]*[theta.d ;theta.X]);
    if(sum(isinf(lamTmp(:))) ==0 && max(lamTmp(:)) <= maxCount)
       flag = 0; 
    end
    
end


end