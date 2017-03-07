function [c, varargout] = confusion_matrix(gt, pred)
% This function outputs the confusion matrix of gt against pred.
assert(numel(gt) == numel(pred));
all_cls = unique([gt(:); pred(:)]);
num_cls = numel(all_cls);
cls_to_ind_map = containers.Map(all_cls, 1:num_cls);
c = zeros(num_cls, num_cls);
for i = 1:numel(gt)
    ind_gt = cls_to_ind_map(gt(i));
    ind_pred = cls_to_ind_map(pred(i));
    c(ind_gt, ind_pred) = c(ind_gt, ind_pred) + 1;
end
disp('Class label to index:');
disp(cls_to_ind_map.keys);
disp(cls_to_ind_map.values);
disp('Confusion Matrix:');
disp(c);
if nargout > 1
    varargout{1} = cls_to_ind_map;
end

end
