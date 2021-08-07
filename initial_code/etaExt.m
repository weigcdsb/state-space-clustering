function eta = etaExt(eta, u, alphaDP)

u_star = min(u);

rho = eta2rho(eta);
while(sum(rho) <= (1- u_star))
    eta_new = betarnd(1, alphaDP);
    eta = [eta eta_new];
    rho = eta2rho(eta);
end

end