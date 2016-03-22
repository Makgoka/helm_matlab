function func = funcparse(func_type)


if strcmp(func_type, 'lin')
    func = @lin;
elseif strcmp(func_type, 'sig')
    func = @sigmoid;
elseif strcmp(func_type, 'relu')
    func = @relu;
elseif strcmp(func_type, 'tanh')
    func = @tanh;
elseif strcmp(func_type, 'mm')
    func = @mm;
else
    error('Not supported function type!');
end

end

function y = sigmoid(x)


y = 1. ./ (1. + exp(-x));

end

function y = lin(x)


y = x;

end

function y = relu(x)

y = x .* (x >= 0);

end

function y = mm(x)

y = x;
y(y>1) = 1;
y(y<-1) = -1;

end

