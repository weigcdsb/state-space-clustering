
    



% update both column- & block-wise
C_fit(:,:,g) = C_fit(:,:,g-1);
    for j = unique(Lab)
        
        latentId = ((j-1)*p+1):(j*p);
        X_tmp = X_fit(latentId,:,g);
        for l = 1:p
            lamc = @(c) exp(C_fit(Lab == j,setdiff(1:end, l),g)*...
                X_tmp(setdiff(1:end, l),:) + d_fit(Lab == l, g) +...
                c*X_tmp(l,:));
            
            derc = @(c) (Y(Lab==j,:) - lamc(c))*X_tmp(l,:)' -...
                inv(Sigcjl0)*(c - mucjl0);
            hessc = @(c) -diag(sum(repmat(X_tmp(l,:).^2,n ,1).*...
                lamc(c),2))-inv(Sigcjl0);
            
            [muc,~,niSigc,~] = newton(derc,hessc,C_fit(Lab == j,l,g-1),1e-6,1000);
            Sigc = -inv(niSigc);
            Sigc = (Sigc + Sigc')/2;
            
%             [muc C_all(Lab ==j, 1)]
            
            C_fit(Lab == j,l,g) = mvnrnd(muc,Sigc)';
            
        end
    end