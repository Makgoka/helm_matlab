% load_data.m

addpath(fullfile(pwd, 'data'));

xTrain = loadMNISTImages('train-images-idx3-ubyte');
xTest  = loadMNISTImages('t10k-images-idx3-ubyte');
yTrain = loadMNISTLabels('train-labels-idx1-ubyte');
yTest  = loadMNISTLabels('t10k-labels-idx1-ubyte');


xTrain = xTrain';
xTest  = xTest';
yTrain(yTrain == 0) = 10;
yTest(yTest == 0) = 10;