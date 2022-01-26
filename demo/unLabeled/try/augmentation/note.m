


% calculate Chebychev parameters
% chebMax = 10*max(Y(:));
% Cheb = zeros(1+chebMax, 2+1);
% for k = 0:chebMax % k = min(Y(:)):max(Y(:))
%     if k ==0
%         cehbTmp = compute_cheb([log(0.1)-2 log(0.1)+2]);
%         %         cehbTmp = compute_cheb([log(0.1) log(1)]);
%     else
%         cehbTmp = compute_cheb([log(k)-2 log(k)+2]);
%         %         cehbTmp = compute_cheb([log(k) log(k+1)]);
%     end
%     Cheb(k+1,:) = cehbTmp;
% end



%             cheb_a = zeros(T,1);
%             cheb_b = zeros(T,1);
%             for t = 1:T
%                 idx = min(round(exp(THETA{g}(cc).d(t))) + 1, chebMax + 1);
%                 cheb_a(t) = Cheb(idx,3);
%                 cheb_b(t) = Cheb(idx,2);
%             end
%             logLike = poiLogMarg_PAL(Y(ii,:)', THETA{g}(cc).X', THETA{g}(cc).d', prior,...
%                 cheb_a, cheb_b);





%         cheb_a = zeros(T,1);
%         cheb_b = zeros(T,1);
%         for t = 1:T
%             idx = min(round(exp(THETA{g}(c_prop).d(t))) + 1, chebMax + 1);
%             cheb_a(t) = Cheb(idx,3);
%             cheb_b(t) = Cheb(idx,2);
%         end
%         logLike = poiLogMarg_PAL(Y(ii,:)', THETA{g}(c_prop).X', THETA{g}(c_prop).d', prior,...
%             cheb_a, cheb_b);




%% MH_notes...



% ues MH
%         derc = @(c) X_tmpC'*(Y(i,:)' - lamC(c)) - prior.SigC0\(c - prior.muC0);
%         hessc = @(c) -X_tmpC'*diag(lamC(c))*X_tmpC - inv(prior.SigC0);
%         c0 = C_fit(i,:,g-1)';
%
%         [muc,~,niSigc,~] = newton(derc,hessc,c0,1e-8,1000);
%         if(sum(isnan(muc)) ~= 0)
%             disp('use 0')
%             [muc,~,niSigc,~] = newton(derc,hessc,zeros(size(c0)),1e-8,1000);
%         end
%
%         % normal approximation
%         Rc = chol(-niSigc,'lower'); % sparse
%         zc = randn(length(muc), 1) + Rc'*muc;
%         C_fit(i,:,g) = Rc'\zc;


