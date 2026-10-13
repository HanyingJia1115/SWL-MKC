function [F,iter,obj,AA] = algorithm(KH,k,W,alpha,theta,yita,tau)

num = size(KH,1);
numker = size(KH,3);
maxIter = 150;

miu = 1.2;
rho = 1e-2;
beta = ones(num,1)/num;
gamma = ones(numker,1)/numker;
lambda = ones(numker,1)/sqrt(numker);
H = zeros(num,k);
F = zeros(num,k);
opt.disp = 0;

viewnum = 2;
A = cell(1,viewnum);
C = cell(1,viewnum);
for iv = 1:viewnum
    A{iv} = zeros(num,k);
    C{iv} = zeros(num,k);
end
A{1} = H;
A{2} = F;

Hp = cell(1,numker);
WHp = cell(1,numker);
WHpR = cell(1,numker);
R = cell(1,numker);
for p = 1:numker
    R{p} = eye(k);
end

%% Kernel preprocessing
K0 = zeros(num,num);
diagKH = zeros(num,numker);
for p = 1:numker
    KH(:,:,p) = (KH(:,:,p) + KH(:,:,p)')/2;
    [Hp0,~] = eigs(KH(:,:,p),k,'la',opt);

    K0 = K0 + (1/numker)*KH(:,:,p);
    Hp{p} = Hp0;
    WHp{p} = W{p}*Hp{p};
    WHpR{p} = WHp{p};       
    diagKH(:,p) = diag(KH(:,:,p));
end

%% Initial fused kernel K_gamma
Kgamma = zeros(num,num);
for p = 1:numker
    Kgamma = Kgamma + gamma(p)^2*KH(:,:,p);
end
Kgamma = (Kgamma + Kgamma')/2;

%% Fixed neighborhoods
numSel = round(tau*num);
NS = genarateNeighborhood(K0,numSel);
% 提前保存 A_beta 的非零元素位置
rowA = NS(:);
colA = repelem((1:num)',numSel);

% These arrays have fixed maximum sizes; trim them before returning.
obj = zeros(1,maxIter+1);
AA = zeros(1,maxIter+1);
KHatmatrix = zeros(num,num,numker);

D0 = zeros(num,k);
for p = 1:numker
    D0 = D0 + lambda(p)*WHpR{p};
end

flag = 1;
iter = 0;

tic
while flag
    iter = iter + 1;
     %% Build Amatrix from the current beta

       betaA = repelem(beta,numSel);

       A_beta = sparse( ...
    rowA,colA,betaA,num,num);

       Amatrix = A_beta*A_beta';

    %% Build the localized kernels used by the gamma update
    for p = 1:numker
        KHatmatrix(:,:,p) = KH(:,:,p).*Amatrix;
        KHatmatrix(:,:,p) = ...
            (KHatmatrix(:,:,p) + KHatmatrix(:,:,p)')/2;
    end

    %% the first step--optimize H
    G = Kgamma.*Amatrix;
    B = G*H + 0.5*(theta*F + rho*A{1} - C{1});
    [u,~,v] = svd(B,'econ');
    H = u*v';
    HHt = H*H';

    %% the second step--optimize F
    D = alpha*D0 + theta*H + rho*A{2} - C{2};
    [U,~,V] = svd(D,'econ');
    F = U*V';

    %% the third step--optimize gamma
    g2 = zeros(numker,1);
    diagA = full(diag(Amatrix));
    for p = 1:numker
        Q1 = diagKH(:,p)'*diagA;
        Q2 = sum(sum(KHatmatrix(:,:,p).*HHt));
        g2(p) = Q1 - Q2;
    end

    invg2 = 1./g2;
    gamma = invg2/sum(invg2);

    % Update Kgamma using the new gamma.
    Kgamma = zeros(num,num);
    for p = 1:numker
        Kgamma = Kgamma + gamma(p)^2*KH(:,:,p);
    end
    Kgamma = (Kgamma + Kgamma')/2;

    %% the fourth step--optimize beta
    e2 = zeros(num,1);
    for i = 1:num
        idx = NS(:,i);
        Ki = Kgamma(idx,idx);
        Hi = H(idx,:);

        N1 = trace(Ki);

        % Equivalent to trace(Hi'*Ki*Hi), but Ki*Hi is formed first.
        KiHi = Ki*Hi;
        N2 = sum(sum(Hi.*KiHi));

        e2(i) = N1 - N2;
    end

    beta = (1./e2)/sum(1./e2);

    %% the fifth step--optimize R_p
    for p = 1:numker
        Mp = lambda(p)*(WHp{p}'*F);
        [Ur,~,Vr] = svd(Mp,'econ');
        R{p} = Ur*Vr';
    end

    %% the sixth step--optimize lambda
    f = zeros(1,numker);
    for p = 1:numker
        WHpR{p} = WHp{p}*R{p};
        f(p) = sum(sum(F.*WHpR{p}));
    end
    lambda = f/norm(f,2);

    % This D0 is used by the objective below and by the next F update.
    D0 = zeros(num,k);
    for p = 1:numker
        D0 = D0 + lambda(p)*WHpR{p};
    end

    %% the seventh step--optimize tensor A, W, rho
    S_tensor = cat(3,H,F);
    C_tensor = cat(3,C{:});
    Sv = S_tensor(:);
    Cv = C_tensor(:);
    [Av,objV] = wshrinkObj(Sv + 1/rho*Cv,yita/rho, ...
                           [num,k,viewnum],0,3);
    AA(iter) = objV;
    A_tensor = reshape(Av,[num,k,viewnum]);
    for iv = 1:viewnum
        A{iv} = A_tensor(:,:,iv);
        C{iv} = C{iv} + rho*(S_tensor(:,:,iv) - A{iv});
    end
    rho = min(miu*rho,1e10);

    %% Objective value
    term1 = sum((beta.^2).*e2);
    term2 = alpha*trace(F'*D0);
    term3 = theta*trace(F'*H);
    term4 = yita*objV;
    obj(iter) = term1 - term2 - term3 + term4;

    if (iter > 3) && ...
            (abs((obj(iter-1)-obj(iter))/obj(iter-1)) < 1e-5 || ...
             iter > maxIter)
        flag = 0;
    end
end
toc

obj = obj(1:iter);
AA = AA(1:iter);

end
