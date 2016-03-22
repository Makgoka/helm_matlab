% elm_ae_l1.m

function [accuracy, elapsed_time] = elm_ae_l1(xTrain, yTrain, xTest, yTest, varargin)
% Use the idea of autoencoder with L1 regularization and ELM classifier,
% which is an implementation of H-ELM.
% train_x, train_y, test_x, test_y should have a number of rows equal to
% num of samples.

%%

th = tic;
%%%%%%%%%%%%% variable settings.
% GAMMA gives the penality weight for ridge regression.
gamma = 1e-1;
% LAMBDA gives the update weight for sparse autoencoder.
lambda = 1e3;
sigmoid = @(x) 1 ./ (1 + exp(-x));
ReLU = @ (x) x.*(x>0);
lin = @(x) x;
% func = @tansig;
numTrainSample = size(xTrain, 1);
inputDim = size(xTrain, 2);
numTestSample = size(xTest, 1);

if nargin == 4
    func = lin;
    % decide how many neurons should be at each hidden layer.
    numNeuron = [200];  % This should be further modified.
    numLastClassifierNeuron  = 2000;
else
    func = varargin{1}.func;
    numNeuron = varargin{1}.numNeuron;
    numLastClassifierNeuron = varargin{1}.numLastClassifierNeuron;
end
numAutoencoder = numel(numNeuron);
% construct the sequence of autoencoders. 
autoEncoderCell = cell(1, numAutoencoder); 

% % generate random paramater matrix used in ELM.
% % THETA is used only once.
% numNeuronWithInput = [inputDim, numNeuron];
% for i = 1:numAutoencoder
%     theta = 2*rand(numNeuronWithInput(i)+1, numNeuronWithInput(i+1))-1;
%     [autoEncoderCell{i}.theta, ~] = qr(theta, 0);      % using reduced QR factorization
%     % autoEncoderCell{i}.theta = 2*rand(inputDim+1, numNeuron(i))-1;
%     autoEncoderCell{i}.numNeuron = numNeuron(i);
% end
% clear theta;


%%%%%%%%%%%%%% forward training in an ELM sense.

% forward propagate through every encoder and determine beta,
% which is the (recover) weight for each output layer in autoencoder.

% initialize the input
currentOutput  = xTrain;
numNeuronWithInput = [inputDim, numNeuron];
for i = 1:numAutoencoder
    % We create theta and discard it at once.
    theta = 2*rand(numNeuronWithInput(i)+1, numNeuronWithInput(i+1))-1;
%     [theta, ~] = qr(theta, 0);      % using reduced QR factorization
    [autoEncoderCell{i}.beta, currentOutput] = ...
    elm_autoencoder(currentOutput, theta, lambda, func);
    currentOutput  = mapminmax(currentOutput')';
end
clear theta;
clear xTrain;


% after the autoencoding, we use whatever classification method for output.
% here a standard elm classifier is applied.
ELM.numNeuron = numLastClassifierNeuron;
ELM.theta = 2*rand(size(currentOutput, 2)+1, ELM.numNeuron)-1;
H = tansig([ones(numTrainSample, 1), currentOutput]*ELM.theta);
opts.SYM = true;
opts.POSDEF = true;
ELM.beta = linsolve(H'*H + eye(size(H, 2))/gamma, H'*yTrain, opts); 
fprintf('\nThe training of the last classifier is finished.\n');
clear H;    % this at least saves some memory.
clear yTrain;

% The training is finished.
fprintf('\nTraining is finished.\n');

%%%%%%%%%%%%% testing 
currentOutput  = xTest;
for i = 1:numAutoencoder
    % currentOutput = func([biasWeight*ones(numTestSample, 1), ...
    %     currentOutput]*autoEncoderCell{i}.theta)*autoEncoderCell{i}.beta';
    currentOutput = func(currentOutput*autoEncoderCell{i}.beta');
    currentOutput  = mapminmax(currentOutput')';
end

currentOutput = tansig([ones(numTestSample, 1), ...
    currentOutput]*ELM.theta)*ELM.beta;

% The maximum gives the position to which class 
% the corresponding sample most probably belongs.
[~, yClassified] = max(currentOutput, [], 2);

% We transform YTEST into the same mode as YCLASSIFY.
yTestCompact= zeros(size(yTest, 1), 1);
for i = 1:size(yTest, 1)
    [~, id] = max(yTest(i, :));
    yTestCompact(i) = id;
end

accuracy = (yTestCompact == yClassified);
accuracy = sum(accuracy) ./ numel(accuracy);

% testing is finished.
fprintf('\nTesting is finished.\n');
fprintf('\nAccuracy: %.2f\n', accuracy * 100);

elapsed_time = toc(th);

fprintf('\nElapsed time: %.4f\n', elapsed_time);
