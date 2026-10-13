function [W, Wvec, D] = constructInstanceWeight(KH, q)
% constructInstanceWeight
% 根据 R-IWLF-MVC 论文构造每个视图的实例权重 W_p
%
% 输入：
%   KH   : n × n × m，多核矩阵
%   q    : 实例权重指数参数
%
% 输出：
%   W    : m × 1 cell，每个 W{p} 是 n × n 对角矩阵
%   Wvec : n × m，每一列是第 p 个视图的实例权重向量
%   D    : n × m，每一列是第 p 个核矩阵的度向量

    num = size(KH,1);
    numker = size(KH,3);

    W = cell(numker,1);
    Wvec = zeros(num,numker);
    D = zeros(num,numker);

    oneVec = ones(num,1);

    for p = 1:numker

        % 保证核矩阵数值对称
        Kp = (KH(:,:,p) + KH(:,:,p)') / 2;

        % D_p = K_p * 1
        dp = Kp * oneVec;

        % 避免中心化核矩阵产生负度或数值误差
        dp = max(real(dp),0);

        % W_p = D_p^(q/2)
        wp = dp .^ (q/2);

        % 约束 sum_i W_pi = n
        if sum(wp) > eps
            wp = num * wp / sum(wp);
        else
            % 若度全部为 0，退化为均匀权重
            wp = ones(num,1);
        end

        D(:,p) = dp;
        Wvec(:,p) = wp;

        % 目标函数中 W_p 作为对角矩阵
        W{p} = spdiags(wp,0,num,num);
    end
end