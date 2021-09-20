function paramsOut = LDStrans(M, params, nClus, p)

M_tilda = blkdiag(1,M);

% latent vectors:
paramsOut.xt = M*params.xt;
paramsOut.x0 = M*params.x0;
paramsOut.Q0 = M*params.Q0*M';
paramsOut.mux00 = M*params.mux00;
paramsOut.Sigx00 = M*params.Sigx00*M';

% loading:
paramsOut.C = params.C/M;

paramsOut.mudc = zeros(size(params.mudc));
paramsOut.Sigdc = zeros(size(params.Sigdc));
paramsOut.deltadc0 = zeros(size(params.deltadc0));
paramsOut.Taudc0 = zeros(size(params.Taudc0));
paramsOut.Psidc0 = zeros(size(params.Psidc0));

% linear dynamics
paramsOut.Psi0 = zeros(size(params.Psi0));

for j = 1:nClus
   idxTmp = id2id(j,p);
   MTmp = M(idxTmp, idxTmp);
   MtildaTmp = blkdiag(1, MTmp);
   
   paramsOut.mudc(:,j) = (MtildaTmp')\params.mudc(:,j);
   paramsOut.Sigdc(:,:,j) = ((MtildaTmp')\params.Sigdc(:,:,j))/MtildaTmp;
   paramsOut.deltadc0(:,j) = (MtildaTmp')\params.deltadc0(:,j);
   paramsOut.Taudc0(:,:,j) = ((MtildaTmp')\params.Taudc0(:,:,j))/MtildaTmp;
   paramsOut.Psidc0(:,:,j) = ((MtildaTmp')\params.Psidc0(:,:,j))/MtildaTmp;
   
   % linear dynamics
   paramsOut.Psi0(:,:,j) = MTmp*params.Psi0(:,:,j)*MTmp';
end


% linear dynamics
paramsOut.A = M*params.A/M;
paramsOut.b = M*params.b;
paramsOut.Q = M*params.Q*M';
paramsOut.BA0_all = ((M_tilda')\params.BA0_all)*M';
paramsOut.Lamb0 = M_tilda*params.Lamb0*M_tilda';


end