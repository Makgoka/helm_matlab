% pre_processing.m


xTrain = bsxfun(@minus, xTrain, mean(xTrain, 2));
xTest  = bsxfun(@minus, xTest, mean(xTest, 2));
xTrain = bsxfun(@rdivide, xTrain, var(xTrain, 0, 2));
xTest  = bsxfun(@rdivide, xTest, var(xTest, 0, 2));

