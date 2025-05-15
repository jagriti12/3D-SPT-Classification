rng(0); % Fixing the random seed for reproducibility
Layers = 10;
k = 5; % Number of folds

%% Split Data into Training and Testing Sets %%
cvpart = cvpartition(y_train, 'Holdout', 0.1);
X_train = X_Combined(training(cvpart), :);
y_train_part = y_train(training(cvpart));
X_test = X_Combined(test(cvpart), :);
y_test = y_train(test(cvpart));

%% Single Tree with CV %%
cvSingleTree = fitctree(X_train, y_train_part, 'CrossVal', 'on', 'KFold', k);
cvSingleTreeLoss = kfoldLoss(cvSingleTree);
predictionsSingleTree = kfoldPredict(cvSingleTree);
accuracySingleTree = sum(predictionsSingleTree == y_train_part) / length(y_train_part);

%% Random Forest with Manual CV %%
cvpartRF = cvpartition(y_train_part, 'KFold', k);
oobErrorRF = zeros(k, 1);

for i = 1:k
    trainingIndices = training(cvpartRF, i);
    testIndices = test(cvpartRF, i);
    modelRF = TreeBagger(100, X_train(trainingIndices, :), y_train_part(trainingIndices, :), ...
                         'Method', 'classification', 'OOBPrediction', 'On', ...
                         'MaxNumSplits', 2^Layers - 1);
    oobErrorRF(i) = oobError(modelRF, 'Mode', 'ensemble');
end

meanOOBErrorRF = mean(oobErrorRF);
accuracyRF = 1 - meanOOBErrorRF;

%% Boosted Trees %%
cvBoostedTrees = fitcensemble(X_train, y_train_part, 'Method', 'RUSBoost', ...
                              'CrossVal', 'on', 'KFold', k, 'NumLearningCycles', 100);
cvBoostedLoss = kfoldLoss(cvBoostedTrees);
predictionsBoosted = kfoldPredict(cvBoostedTrees);
accuracyBoosted = sum(predictionsBoosted == y_train_part) / length(y_train_part);

%% Bagged Trees %%
cvBaggedTrees = fitcensemble(X_train, y_train_part, 'Method', 'Bag', ...
                             'CrossVal', 'on', 'KFold', k, 'NumLearningCycles', 100);
cvBaggedLoss = kfoldLoss(cvBaggedTrees);
predictionsBagged = kfoldPredict(cvBaggedTrees);
accuracyBagged = sum(predictionsBagged == y_train_part) / length(y_train_part);

%% Print out accuracies for comparison %%
fprintf('Accuracy of Single Decision Tree: %.2f%%\n', accuracySingleTree * 100);
fprintf('Accuracy of Random Forest: %.2f%%\n', accuracyRF * 100);
fprintf('Accuracy of Boosted Trees: %.2f%%\n', accuracyBoosted * 100);
fprintf('Accuracy of Bagged Trees: %.2f%%\n', accuracyBagged * 100);

%% Print out losses for comparison %%
fprintf('Cross-Validated Loss of Single Decision Tree: %.2f%%\n', cvSingleTreeLoss * 100);
fprintf('Mean OOB Error of Random Forest: %.2f%%\n', meanOOBErrorRF * 100);
fprintf('Cross-Validated Loss of Boosted Trees: %.2f%%\n', cvBoostedLoss * 100);
fprintf('Cross-Validated Loss of Bagged Trees: %.2f%%\n', cvBaggedLoss * 100);

%% Store the CV results and accuracies in a structured array %%
cvResults = struct('Model', {'Single Decision Tree', 'Random Forest', 'Boosted Trees', 'Bagged Trees'}, ...
                   'CVLoss', {cvSingleTreeLoss, meanOOBErrorRF, cvBoostedLoss, cvBaggedLoss}, ...
                   'Accuracy', {accuracySingleTree, accuracyRF, accuracyBoosted, accuracyBagged});

%% Analyze the results to find the model with the lowest cross-validation loss %%
[~, bestLossIndex] = min([cvResults.CVLoss]);
bestModelByLoss = cvResults(bestLossIndex);

%% Analyze the results to find the model with the highest accuracy %%
[~, bestAccuracyIndex] = max([cvResults.Accuracy]);
bestModelByAccuracy = cvResults(bestAccuracyIndex);

%% Displaying the Best Models based on Loss and Accuracy %%
fprintf('The best model based on cross-validation loss is: %s with a loss of %.2f%%\n', ...
        bestModelByLoss.Model, bestModelByLoss.CVLoss * 100);
fprintf('The best model based on accuracy is: %s with an accuracy of %.2f%%\n', ...
        bestModelByAccuracy.Model, bestModelByAccuracy.Accuracy * 100);

%% Determine the best model based on the given criteria %%
if bestLossIndex == bestAccuracyIndex
    fprintf('Overall Best Model considering both Loss and Accuracy: %s\n', bestModelByAccuracy.Model);
else
    fprintf('A decision needs to be made between Loss and Accuracy criteria as they suggest different best models.\n');
end

%% Plotting %%
figure;
x = 1:numel(cvResults);  % X locations of the bars
width = 0.3;

% Plotting CV Loss and Accuracy
lossValues = [cvResults.CVLoss] * 100; % Convert to percentage
accuracyValues = [cvResults.Accuracy] * 100; % Convert to percentage

yyaxis left;
plot(x, lossValues, '-o');
ylabel('Cross-Validated Loss (%)', 'Color', 'black');
ylim([0, max(lossValues) + 10]);
set(gca, 'YColor', 'black');

yyaxis right;
plot(x, accuracyValues, '-s');
ylabel('Accuracy (%)', 'Color', 'black');
ylim([0, 100]);
set(gca, 'YColor', 'black');

set(gca, 'XTick', x, 'XTickLabel', {cvResults.Model}, 'XTickLabelRotation', 45, ...
    'XColor', 'black', 'YColor', 'black');

% Setting the tick label colors
set(gca, 'TickLabelInterpreter', 'none', ...
    'XTickLabel', get(gca, 'XTickLabel'), 'YTickLabel', get(gca, 'YTickLabel'), ...
    'TickLabelInterpreter', 'none', ...
    'XTickLabelRotation', 45);

xlabel('Model', 'Color', 'black');
title('Model Performance: CV Loss vs Accuracy');
legend({'CV Loss', 'Accuracy'}, 'Location', 'Best');
grid on;
%% Train the Best Model on the Entire Training Data %%
bestModelIndex = bestAccuracyIndex; % Change this if you want to prioritize CV loss over accuracy
bestModel = cvResults(bestModelIndex);

bestModelTrained = fitcensemble(X_train, y_train_part, 'Method', 'Bag', 'NumLearningCycles', 100);
[y_pred, scores] = predict(bestModelTrained, X_test);

%% Plot the Confusion Matrix %%
figure;
confMat = confusionmat(y_test, y_pred);
confusionchart(confMat);
title('Confusion Matrix');

%% Plot ROC Curves and Calculate AUC %%
colors = lines(numel(unique(y_test))); % This generates a colormap with a distinct color for each class
num_classes = numel(unique(y_test));
legend_labels = cell(num_classes, 1);
TPR_all = cell(num_classes, 1);
FPR_all = cell(num_classes, 1);
AUC_values = zeros(num_classes, 1);

for i = 1:num_classes
    scores_current = scores(:, i);
    true_labels = (y_test == i);
    
    [~, idx] = sort(scores_current, 'descend');
    sorted_labels = true_labels(idx);
    
    TP = cumsum(sorted_labels);
    FP = cumsum(~sorted_labels);
    
    TPR_comp = TP / sum(sorted_labels);
    FPR_comp = FP / sum(~sorted_labels);
    
    TPR_all{i} = TPR_comp;
    FPR_all{i} = FPR_comp;
    
    AUC = trapz(FPR_comp, TPR_comp);
    AUC_values(i) = AUC;
    
    legend_labels{i} = sprintf('Class %d (AUC = %.4f)', i, AUC);

end

figure;
hold on;
for i = 1:num_classes
    plot(FPR_all{i}, TPR_all{i}, 'Color', colors(i, :), 'LineWidth', 2);
end
legend(legend_labels, 'Location', 'Best');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves by Class');
grid on;
hold off;
