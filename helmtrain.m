function hElmModel = helmtrain(xTrain, yTrain, ...
    numNeuron, numELMNeuron, varargin)
% HELMTRAIN trains the helm model for testing.
% Use the idea of autoencoder with L1 regularization and ELM classifier.
% train_x, train_y, test_x, test_y should have a number of rows equal to
% num of samples.
% numNeuron is a vector that the length is the number of layers for the
% autoencoder, and each entry is the number of neurons.
% lambda is the weight term for autoencoders.
% gamma is the regularity weight for ELM.

fprintf('\nBegin training H-ELM model.\n');
%%%%%%%%%%%%% variable settings.

numTrainSample = size(xTrain, 1);
inputDim = size(xTrain, 2);

% r1 = 0.5;
% r2 = 0.1;
% thresh = 0; % threshold for the 2-class classification
use_gpu = false;

opts.func = 'relu';
opts.elm_func = 'tansig';
opts.ae_weight = 1e2;
opts.elm_weight = 1e-1;
opts.r1 = 0.5;
opts.r2 = 0.1;
opts.use_gpu = false;
opts.num_ae_neuron = numNeuron;
opts.num_autoencoder = numel(numNeuron);
opts.num_elm_neuron = numELMNeuron;
opts.verbose = true;
opts = vl_argparse(opts, varargin);

func = opts.func;
elm_func = opts.elm_func;
lambda = opts.ae_weight;
gamma = opts.elm_weight;
r1 = opts.r1;
r2 = opts.r2;
verbose = opts.verbose;

% numAutoencoder gives the number of layers for the autoencoder
numAutoencoder = numel(numNeuron);
if numAutoencoder == 0
    warning('Not using autoencoders, use ELM only.');
end


ae = struct('beta', []);
autoencoder = repmat(ae, 1, numAutoencoder+1);




%%%%%%%%%%%%%% forward training in an ELM sense.

% forward propagate through every encoder and determine beta,
% which is the (recover) weight for each output layer in autoencoder.
% tic toc is not supported by codegen.
if verbose
    th = tic;
end


% initialize the input
xTrainWithBias = [r2*ones(size(xTrain, 1), 1), xTrain];
currentOutput  = xTrainWithBias;

if numel(numNeuron) > 0 && numNeuron(1) > 0 
    % if we use autoencoder and the number of neurons are not 0

    for i = 1:numAutoencoder
        theta = r1*(2*rand(size(currentOutput, 2), numNeuron(i))-1);
        H = activation(currentOutput*theta, func);
        autoencoder(i).beta = elm_autoencoder(H, currentOutput, lambda, verbose);
        autoencoder(i).theta = theta;
        currentOutput = activation(currentOutput*autoencoder(i).beta', func);
    end
end

% after the autoencoding, we use whatever classification method for output.
% here a standard elm classifier is applied.
ELM = struct('numELMNeuron', 0, 'theta', [], 'beta', []);
coder.varsize('ELM.theta', 'ELM.beta');
ELM.numELMNeuron = numELMNeuron;
ELM.theta = r1*(2*rand(size(currentOutput, 2), ELM.numELMNeuron)-1);
H = activation(currentOutput*ELM.theta, elm_func);



if use_gpu
    H_gpu = gpuArray(H);
    yTrain_gpu = gpuArray(yTrain);
    lhs = H_gpu'*H_gpu + eye(size(H, 2), 'gpuArray')/gamma;
    rhs = H'*yTrain_gpu;
    beta = lhs \ rhs;
    ELM.beta = gather(beta);
else
    linsolve_opts.SYM = true;
    linsolve_opts.POSDEF = true;
    ELM.beta = linsolve(H'*H + eye(size(H, 2))/gamma, H'*yTrain, linsolve_opts); 
end

if verbose
    fprintf('Training of the ELM classifier is finished.\n');
end


% The training is finished.
if verbose
    fprintf('Elapsed time: %.2f\n', toc(th));
end

hElmModel.numNeuron = numNeuron;
hElmModel.numAutoencoder = numAutoencoder;
hElmModel.autoencoder = autoencoder;
hElmModel.ELM = ELM;
% hElmModel.func = func;
% hElmModel.verbose = verbose;
% hElmModel.r2 = r2;
% hElmModel.thresh = thresh;
hElmModel.opts = opts;