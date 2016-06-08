function [yClassified, statistics] = helmtest(model, xTest, yTest, varargin)


% Assume xTest \in R^{N x p}, where N is the number of test samaples.

numTestSample = size(xTest, 1);


% parse function
opts = model.opts;
opts.thresh = 0;
opts = vl_argparse(opts, varargin);

% func = opts.func;
verbose = opts.verbose;
% r2 = opts.r2;
thresh = opts.thresh;
%%%%%%%%%%%%% testing 

if verbose
	th = tic();
    fprintf('\nTest for H-ELM begins\n');
end

statistics = [];
if size(yTest, 2) == 1
    scores = compute_score(model, xTest);
    for threshold = thresh
        yClassified = scores;
        yClassified(scores > threshold) = 1;
        yClassified(scores <= threshold) = -1;
        truePos  = sum((yClassified == 1) & (yTest == 1));
        falsePos = sum((yClassified == 1) & (yTest == -1));
        falseNeg = sum((yClassified == -1) & (yTest == 1));
        trueNeg  = sum((yClassified == -1) & (yTest == -1));
        accuracy = (truePos+trueNeg)/numel(yClassified);
        probClassification = truePos / (truePos+falseNeg);
        falseAlarm = falsePos / (falsePos+trueNeg);
        if verbose
            fprintf('Threshold: %.4f\n', threshold);
            fprintf('Accuracy: %.4f\n', accuracy);
            fprintf('Probability of Classification: %.4f\n', probClassification);
            fprintf('False Alarm: %.4f\n', falseAlarm);
            fprintf('Elapsed time: %.2fs.\n', toc(th));
        end
    statistics = [statistics; threshold, probClassification, falseAlarm, accuracy];
    end
else
    yClassified = -ones(size(yTest));
    [scores, pos] = compute_score(model, xTest);
    trueCount = 0;
    for i = 1:numel(pos)
        yClassified(i, pos(i)) = 1;
        if yTest(i, pos(i)) == 1
            trueCount = trueCount + 1;
        end
    end
    accuracy = trueCount / numel(pos);
    if verbose
        fprintf('Accuracy: %.4f\n', accuracy);
        fprintf('Elapsed time: %.2fs.\n', toc(th));
    end
end




end