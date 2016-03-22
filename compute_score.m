function [scores, varargout] = compute_score(model, xTest, varargin)

% parse function
opts = model.opts;
opts = vl_argparse(opts, varargin);

func = opts.func;
% verbose = opts.verbose;
r2 = opts.r2;
elm_func = opts.elm_func;
% thresh = opts.thresh;
%%%%%%%%%%%%% testing 

% This is the part for autoencoder

currentOutput = [r2*ones(size(xTest, 1), 1), xTest];
% H = activation(currentOutput*model.autoencoder(1).beta', func);

if numel(model.numNeuron) > 0 && model.numNeuron(1) > 0
for i = 1:model.numAutoencoder
%     H = activation(currentOutput*model.autoencoder(i).beta', func);
%     currentOutput = H*model.autoencoder(i+1).beta;
    currentOutput = activation(currentOutput*model.autoencoder(i).beta', func);
end
end
% H = activation(currentOutput*model.autoencoder(end-1).beta', func);
% currentOutput = H*model.autoencoder(end).beta;

% This is the part for the ELM
H = activation(currentOutput*model.ELM.theta, elm_func);
% H = activation(currentOutput*model.ELM.theta, func);
% currentOutput = H*model.ELM.beta;
output = H*model.ELM.beta;

if size(output, 2) == 1
    scores = output;
else
    [scores, pos] = max(output, [], 2);
end

if nargout > 1
    varargout{1} = pos;
end