function y = activation(x, func_type)


% Switch between different flags and return directly function value.
% This prevents using function handle to pass between functions.

if strcmp(func_type, 'lin')
    y = x;
elseif strcmp(func_type, 'sig')
    y = 1 ./ (1 + exp(-x));
elseif strcmp(func_type, 'relu')
    y = x .* (x >= 0);
elseif strcmp(func_type, 'tanh')
    y = tanh(x);
elseif strcmp(func_type, 'tansig')
    y = tansig(x);
elseif strcmp(func_type, 'bl')  % bounded linear
    y = 2*x;
    y(y>1) = 1;
    y(y<-1) = -1;
elseif strcmp(func_type, 'lrelu')
    y = x.*(x>=0)+0.1*x.*(x<0);
elseif strcmp(func_type, 'mapminmax')   % mapminmax
    [y, ~] = mapminmax(x, -1, 1);
else
    error('Not supported function type!');
end

end