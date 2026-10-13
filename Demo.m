clear;
clc;
close all;

dataNames{1}  = 'SUNRGBD_fea';
dataNames{2}  = 'caltech101_nTrain10_48';
dataNames{3}  = 'caltech101_nTrain15_48';
dataNames{4}  = 'caltech101_nTrain20_48';
dataNames{5}  = 'caltech101_nTrain25_48';
dataNames{6}  = 'caltech101_nTrain30_48';
dataNames{7}  = 'CCV';
dataNames{8}  = 'ORL';
dataNames{9}  = 'flower17_numberOf_170';
dataNames{10} = 'YALE';
dataNames{11} = 'plant';
dataNames{12} = 'mfeat';
dataNames{13} = 'AR10P';
dataNames{14} = 'bbcsport2view';
dataNames{15} = 'flower102';
dataNames{16} = 'proteinfold';
dataNames{17} = 'UCI_DIGIT';
dataNames{18} = 'nonpl';
dataNames{19} = 'CCV';

scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    codePath = pwd;
else
    codePath = fileparts(scriptPath);
end
addpath(genpath(codePath));
dataPath = fullfile(codePath, 'datasets');

datasetIndices = 10;  

alphaSet = 2.^(-1:7);
thetaSet = 2.^(-1:7);
yitaSet  = 2.^(-1:7);
tau = 0.1;
qnorm = 2;

for i = datasetIndices
    dataName = dataNames{i};
    dataFile = fullfile(dataPath, [dataName, '_Kmatrix.mat']);

    if ~isfile(dataFile)
        error('找不到数据文件：%s', dataFile);
    end

    data = load(dataFile, 'KH', 'Y');
    KH = data.KH;
    Y = data.Y(:);

    numClass = numel(unique(Y));
    numKernel = size(KH, 3);
    numSample = size(KH, 1);

    fprintf('\n数据集：%s\n', dataName);
    fprintf('样本数：%d，核数量：%d，聚类数：%d\n\n', ...
        numSample, numKernel, numClass);

    KH = kcenter(KH);
    KH = knorm(KH);
    W = constructInstanceWeight(KH, qnorm);

    bestACC = -inf;
    bestResult = [];
    experiment = 0;
    totalExperiments = numel(alphaSet) * numel(thetaSet) * numel(yitaSet);

    for alpha = alphaSet
        for theta = thetaSet
            for yita = yitaSet
                experiment = experiment + 1;

                F = algorithm(KH, numClass, W, alpha, theta, yita, tau);
                result = myNMIACCwithmean(F, Y, numClass);

                ACC = result(1);
                NMI = result(2);
                purity = result(3);
                ACCStd = result(4);
                NMIStd = result(5);
                purityStd = result(6);

                if ACC > bestACC
                    bestACC = ACC;
                    bestResult = [alpha, theta, yita, ...
                        ACC, NMI, purity, ACCStd, NMIStd, purityStd];
                end

                fprintf(['%d/%d  alpha=%.4g  theta=%.4g  yita=%.4g  ' ...
                    'ACC=%.4f+-%.4f  NMI=%.4f+-%.4f  ' ...
                    'Purity=%.4f+-%.4f  BestACC=%.4f\n'], ...
                    experiment, totalExperiments, alpha, theta, yita, ...
                    ACC, ACCStd, NMI, NMIStd, purity, purityStd, bestACC);
            end
        end
    end

    fprintf('\n%s 参数搜索完成\n', dataName);
    fprintf('最优参数：alpha=%.4g  theta=%.4g  yita=%.4g\n', ...
        bestResult(1), bestResult(2), bestResult(3));
    fprintf(['最优结果：ACC=%.4f+-%.4f  NMI=%.4f+-%.4f  ' ...
        'Purity=%.4f+-%.4f\n'], ...
        bestResult(4), bestResult(7), bestResult(5), bestResult(8), ...
        bestResult(6), bestResult(9));
end
