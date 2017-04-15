function v = one_hot(y, num_classes)
% Create `one-hot` labels for y, where positive labels are 1 and negative
% labels are -1.
% The labels of y is either from 0 to num_classes -1, where 0 is mapped to
% num_classes, or 1 to num_classes.

v = -ones(numel(y), num_classes);
y(y==0) = num_classes;
for i = 1:numel(y)
    v(i, y(i)) = 1;
end

end