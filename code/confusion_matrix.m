function confusionMatrix = confusion_matrix(groundtruth , prediction)
% if confusionmat are supported use it then
category = unique(groundtruth);

confusionMatrix = zeros(numel(category) , numel(category)) ;
for i = 1:numel(category)
    index = find(groundtruth == category(i)) ; 
    prediction_labels = prediction(index);
    
    for j = 1:numel(category)
        confusionMatrix(i,j) = numel(find(prediction_labels == j));
    end
end
confusionMatrix = 255*confusionMatrix/100 ;
