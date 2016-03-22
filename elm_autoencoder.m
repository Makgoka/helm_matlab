% elm_autoencoder.m

function betaBatch = elm_autoencoder(H, X, lambda, verbose)

% [BETA] = ELM_AUTOENCODER(X, THETA, lambda) gives the autoencoder calculation results.
% More specifically, it solves the following opt problem:
% min_{beta} 1/2||g(X*theta)*beta-X||^2_2 + lambda*||beta||_1 (or 0).
% Solving this problem requires the IRLS scheme.
%
% X is in \mathbb{R}^{N x (p+1)}, where N is the number of samples and p the dimensionality.
% THETA should be in \mathbb{R}^{(p+1) x K}, containing the bias term.  K is the dimensionality of the next layer.
% BETA is in \mathbb{R}^{K x p}.
% LAMBDA is the regularity parameter.




 
% Initialization. 
inputDim  = size(X, 2);
% hiddenDim = size(theta, 1);
hiddenDim = size(H, 2);
betaBatch = zeros(hiddenDim, inputDim); % R^{lx(p+1)}
HtransH   = H'*H;
HtransX   = H'*X;
opts.SYM = true;
opts.POSDEF = true;
iterNum = 0;
% Thresholding and stopping rule.
maxIter = 50;
thresholdUpdate = 1e-1;
thresholdStop   = 1e-3;

% We compute beta column-wise, with the IRLS strategy.
for i = 1:inputDim
    k = 0;
    diagWeight = ones(hiddenDim, 1); 
    xPrevious = zeros(hiddenDim, 1);
    x = ones(hiddenDim, 1);
    while k < maxIter
        k = k + 1;
        lhs = HtransH + diag(2*lambda./diagWeight);
        x = linsolve(lhs, HtransX(:, i), opts);
        diagWeight = abs(x) + thresholdUpdate;
        if sqrt(sum((x-xPrevious).^2)) < thresholdStop
            break;
        end
        xPrevious = x;
    end
    betaBatch(:, i) = x;
    if k > iterNum
        iterNum = k;
    end
end

if verbose
    fprintf('Max Iteration in Autoencoder: %d.\n', iterNum);
end


end
