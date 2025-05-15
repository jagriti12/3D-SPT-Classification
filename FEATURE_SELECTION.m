%% Setup for reproducibility
rng(0);
tic

% Initialize the parallel pool
if isempty(gcp('nocreate'))
    parpool('local');
end

% Assuming train_ftrs and train_lbls are already defined and loaded in the workspace
combined_train_ftrs_lbls = vertcat(train_ftrs, train_lbls);
combined_train_ftrs_lbls = transpose(combined_train_ftrs_lbls);
combined_test_ftrs_lbls = vertcat(test_ftrs, test_lbls);
combined_test_ftrs_lbls = transpose(combined_test_ftrs_lbls);

%% Define Variable Names for Features and Label
variableNames = {'Alpha', 'AngularGaussianityIndex', 'Asymmetry', 'AvgMSDRatio', 'Efficiency',...
                 'FractalDimension', 'Gaussianity', 'JumpLength', 'Kurtosis', 'MaximalExcursion',...
                 'MeanMaximalExcursion', 'Straightness', 'Trappedness', 'VelocityAutocorrelation','Labels'};
labelColumnName = 'Labels';

%% Convert Arrays to Tables with Appropriate Variable Names
T_features_train = array2table(combined_train_ftrs_lbls(:, 1:end-1), 'VariableNames', variableNames(1:end-1));
T_features_train.(labelColumnName) = combined_train_ftrs_lbls(:, end);

T_features_test = array2table(combined_test_ftrs_lbls(:, 1:end-1), 'VariableNames', variableNames(1:end-1));
T_features_test.(labelColumnName) = combined_test_ftrs_lbls(:, end);

% Initialize results storage
results = struct('Method', [], 'Accuracy', [], 'Model', [], 'Features', []);

% Create a waitbar
wb = waitbar(0, 'Initializing...');

%% MRMR Feature Selection
waitbar(0.1, wb, 'Performing MRMR feature selection...');
[idxMRMR, scoresMRMR] = fscmrmr(table2array(T_features_train(:, variableNames(1:end-1))), T_features_train.(labelColumnName));
waitbar(0.2, wb, 'MRMR feature selection completed.');
%% 

% Select top 5 features
topFeaturesMRMR = idxMRMR(1:3);
%% Normalization and Plotting for MRMR
normalizedScoresMRMR = scoresMRMR / max(scoresMRMR);
figure;
bar(normalizedScoresMRMR(idxMRMR));
title('Normalized MRMR Feature Importance Scores');
xlabel('Feature Index');
ylabel('Normalized Importance Score');
xticks(1:length(normalizedScoresMRMR));
xticklabels(variableNames(idxMRMR));
xtickangle(45);

%% NCA Feature Selection
waitbar(0.3, wb, 'Performing NCA feature selection...');
mdlNCA = fscnca(table2array(T_features_train(:, variableNames(1:end-1))), T_features_train.(labelColumnName), 'Solver', 'sgd', 'Standardize', true);
waitbar(0.4, wb, 'NCA feature selection completed.');
%% 

% Select top 5 features
[~, sortedIndices] = sort(mdlNCA.FeatureWeights, 'descend');
topFeaturesNCA = sortedIndices(1:3);
%% Normalization and Plotting for NCA
normalizedFeatureWeights = mdlNCA.FeatureWeights / max(mdlNCA.FeatureWeights);
figure;
bar(normalizedFeatureWeights);
title('Normalized NCA Feature Weights');
xlabel('Feature Index');
ylabel('Normalized Weight');
xticks(1:length(normalizedFeatureWeights));
xticklabels(variableNames(1:end-1));
xtickangle(45);

%% ReliefF Feature Selection
waitbar(0.5, wb, 'Performing ReliefF feature selection...');
[idxReliefF, weightsReliefF] = relieff(table2array(T_features_train(:, variableNames(1:end-1))), T_features_train.(labelColumnName), 10);
waitbar(0.6, wb, 'ReliefF feature selection completed.');
%% 

% Select top 5 features
topFeaturesReliefF = idxReliefF(1:3);
%% Normalization and Plotting for ReliefF
normalizedWeightsReliefF = weightsReliefF / max(weightsReliefF);
figure;
bar(normalizedWeightsReliefF(idxReliefF));
title('Normalized ReliefF Feature Importance Weights');
xlabel('Feature Index');
ylabel('Normalized Importance Weight');
xticks(1:length(normalizedWeightsReliefF));
xticklabels(variableNames(idxReliefF));
xtickangle(45);

%% Combine Top Features from Different Methods for the Combined Model
combinedTopFeatures = unique([topFeaturesMRMR(:); topFeaturesNCA(:); topFeaturesReliefF(:)]);

% Calculate occurrences of each feature
allTopFeatures = [topFeaturesMRMR(:); topFeaturesNCA(:); topFeaturesReliefF(:)];
featureCounts = zeros(1, numel(variableNames)-1);
for i = 1:numel(allTopFeatures)
    featureCounts(allTopFeatures(i)) = featureCounts(allTopFeatures(i)) + 1;
end

% Create a bar plot for the occurrences of each feature
figure;
bar(1:numel(variableNames)-1, featureCounts);
xlabel('Feature Index');
ylabel('Number of Occurrences in Top 3');
title('Occurrences of Each Feature in Top 3 of Each Algorithm');
xticks(1:numel(variableNames)-1);
xticklabels(variableNames(1:end-1));
xtickangle(45);
ylim([0, max(featureCounts) + 1]);
grid on;
%% 

% Define training and test labels for convenience
y_train = T_features_train.(labelColumnName);
y_test = T_features_test.(labelColumnName);

% Define k-fold cross-validation parameter
k = 5;

%% MRMR Features Model with Random Forest
% waitbar(0.2, wb, 'Training and evaluating Random Forest model with MRMR features...');
X_MRMR = table2array(T_features_train(:, variableNames(topFeaturesMRMR)));
accuracies(1) = ensembleCrossValidation(X_MRMR, y_train, k, 100, 10);
results(1).Method = 'MRMR';
results(1).Accuracy = accuracies(1);
results(1).Features = variableNames(topFeaturesMRMR);
fprintf('Cross-Validated Accuracy with MRMR Top Features (RF): %.2f%%\n', accuracies(1) * 100);

%% NCA Features Model with Random Forest
% waitbar(0.4, wb, 'Training and evaluating Random Forest model with NCA features...');
X_NCA = table2array(T_features_train(:, variableNames(topFeaturesNCA)));
accuracies(2) = ensembleCrossValidation(X_NCA, y_train, k, 100, 10);
results(2).Method = 'NCA';
results(2).Accuracy = accuracies(2);
results(2).Features = variableNames(topFeaturesNCA);
fprintf('Cross-Validated Accuracy with NCA Top Features (RF): %.2f%%\n', accuracies(2) * 100);

%% ReliefF Features Model with Random Forest
% waitbar(0.6, wb, 'Training and evaluating Random Forest model with ReliefF features...');
X_ReliefF = table2array(T_features_train(:, variableNames(topFeaturesReliefF)));
accuracies(3) = ensembleCrossValidation(X_ReliefF, y_train, k, 100, 10);
results(3).Method = 'ReliefF';
results(3).Accuracy = accuracies(3);
results(3).Features = variableNames(topFeaturesReliefF);
fprintf('Cross-Validated Accuracy with ReliefF Top Features (RF): %.2f%%\n', accuracies(3) * 100);

%% Combined Features Model with Random Forest
waitbar(1.0, wb, 'Training and evaluating Random Forest model with combined features...');
X_Combined = table2array(T_features_train(:, combinedTopFeatures));
accuracies(4) = ensembleCrossValidation(X_Combined, y_train, k, 100, 10);
results(4).Method = 'Combined';
results(4).Accuracy = accuracies(4);
results(4).Features = variableNames(combinedTopFeatures);
fprintf('Cross-Validated Accuracy with Combined Top Features (RF): %.2f%%\n', accuracies(4) * 100);

% Save the results
save('feature_selection_results.mat', 'results');
toc

% Close the waitbar
close(wb);

% Close the parallel pool
delete(gcp('nocreate'));

%% Ensemble Cross-Validation Function
function cvAccuracy = ensembleCrossValidation(X, y, k, numTrees, numLayers)
    cv = cvpartition(y, 'KFold', k);
    accuracies = zeros(k, 1);
    parfor i = 1:k % Use parfor for parallel processing
        X_train = X(training(cv, i), :);
        y_train = y(training(cv, i));
        X_val = X(test(cv, i), :);
        y_val = y(test(cv, i));
        accuracy = trainAndEvaluateRF(X_train, y_train, X_val, y_val, numTrees, numLayers);
        accuracies(i) = accuracy;
    end
    cvAccuracy = mean(accuracies);
end

% Function to train and evaluate a Random Forest model
function accuracy = trainAndEvaluateRF(X_train, y_train, X_test, y_test, numTrees, numLayers)
    t = templateTree('MaxNumSplits', 2^numLayers - 1);
    mdl = fitcensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', numTrees, 'Learners', t);
    y_pred = predict(mdl, X_test);
    accuracy = sum(y_pred == y_test) / length(y_test);
end
