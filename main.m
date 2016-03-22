% main.m


clear;

load_data;      % This calls the functions that load xTrain, xTest, 
                % yTrain, yTest.
rng(mod(tic, 2^32));
perm1 = randperm(60000, 6000);
perm2 = randperm(10000, 1000);

xTrain = xTrain(perm1, :);
yTrain = yTrain(perm1);
xTest  = xTest(perm2, :);
yTest  = yTest(perm2);

yTrainVec = -ones(size(yTrain, 1), 10);
for i = 1:numel(yTrain)
    yTrainVec(i, yTrain(i)) = 1;
end
yTrain = yTrainVec;

yTestVec = -ones(size(yTest, 1), 10);
for i = 1:numel(yTest)
    yTestVec(i, yTest(i)) = 1;
end
yTest = yTestVec;

pre_processing;


% func = 'relu';  % relu
numNeuron = [500, 100];
numLastClassifierNeuron = 2000;
% gamma = 1e-1;   % gamma is the regularization term in ELM
% lambda =1e2;    % lambda is for autoencoder
% verbose = true;



% [accuracy, elapsed_time] = elm_ae_l1(xTrain, yTrainVec, xTest, yTestVec, options);

% hElmModel = helmtrain(xTrain, yTrain, options);
% [yClassified, accuracy, scores] = helmtest(hElmModel,...
%     xTest, yTest, options);
hElmModel = helmtrain(xTrain, yTrain, numNeuron, numLastClassifierNeuron);
helmtest(hElmModel, xTest, yTest);
