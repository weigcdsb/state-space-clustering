function prior = make_prior(p)

prior.Q0 = eye(p)*0.5^2;
prior.mux00 = zeros(p, 1);
prior.Sigx00 = eye(p);

prior.deltadc0 = [0;ones(p,1)];
prior.Taudc0 = eye(p+1);

prior.Psidc0 = eye(p+1)*1e-2;
prior.nudc0 = p+1+2;

prior.BA0 =[0 1]';
prior.Lamb0 = eye(2);
prior.Psi0 = 1e-2;
prior.nu0 = 1+2;

end