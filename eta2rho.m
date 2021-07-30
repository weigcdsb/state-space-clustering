function rho = eta2rho(eta)

rho = zeros(size(eta));
rho(1) = eta(1);
if(length(rho) > 1)
    for k = 2:length(rho)
        rho(k) = prod(1 - eta(1:(k-1)))*eta(k);
    end
end

end
