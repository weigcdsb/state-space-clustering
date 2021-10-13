function lpdf = iwishlpdf(x, Psi, nu)
pw = size(x,1);
s1 = 0.5*nu*log(det(Psi)) - 0.5*trace(Psi/x) -...
    0.5*nu*pw*log(2) - 0.25*pw*(pw-1)*log(pi) - 0.5*(nu+pw+1)*log(det(x));
s2 = 0;
for kk = 1:pw
   s2 = s2 + gammaln(0.5*(nu-kk+1)); 
end
lpdf = s1-s2;
end




