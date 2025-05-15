rng(0); % Set random number generator seed for reproducibility
tic;

% Assuming these variables are already defined
X_Combined; % Combined feature set
y_train; % Labels

% Split Data into Training and Testing Sets
cvpart_holdout = cvpartition(y_train, 'Holdout', 0.1);
X_train = X_Combined(training(cvpart_holdout), :);
y_train_part = y_train(training(cvpart_holdout));
X_test = X_Combined(test(cvpart_holdout), :);
y_test = y_train(test(cvpart_holdout));

% Number of folds for cross-validation
k = 5;
cvpart = cvpartition(y_train_part, 'KFold', k);

% Define the range for each hyperparameter
vars = [
    optimizableVariable('NumLearningCycles', [150, 500], 'Type', 'integer');
    optimizableVariable('MinLeafSize', [20, 50], 'Type', 'integer')
];

% Bayesian optimization options
hyperopts = struct('Optimizer', 'bayesopt', 'ShowPlots', true, 'CVPartition', cvpart, ...
                   'AcquisitionFunctionName', 'expected-improvement-plus', 'MaxObjectiveEvaluations', 100, ...
                   'UseParallel', true, 'Verbose', 1, 'Repartition', true);

% Define function handle for objective function
objFcn = @(params)bayesObjective(params, X_train, y_train_part, cvpart);

% Running the Bayesian optimization
Results = bayesopt(objFcn, vars, 'MaxObjectiveEvaluations', hyperopts.MaxObjectiveEvaluations);

% Evaluate the optimized model
bestHyperparameters = bestPoint(Results);
BaggedBayesianModel = fitcensemble(X_train, y_train_part, 'Method', 'Bag', ...
    'NumLearningCycles', bestHyperparameters.NumLearningCycles);

predictions = predict(BaggedBayesianModel, X_test);
optimizedAccuracy = sum(predictions == y_test) / length(y_test);
fprintf('Optimized Accuracy: %.2f%%\n', optimizedAccuracy * 100);

% Plot confusion matrix
figure;
confMat = confusionchart(y_test, predictions);
confMat.Title = 'Confusion Matrix';

toc;

[y_pred, scores] = predict(BaggedBayesianModel, X_test);

% Initialize colors for plotting
colors = lines(numel(unique(y_test))); % This generates a colormap with a distinct color for each class

% Plot the Confusion Matrix
figure;
confMat = confusionmat(y_test, y_pred);
confusionchart(confMat);
title('Confusion Matrix');

% Number of classes
num_classes = numel(unique(y_test));

% Prepare to store ROC data and AUC values
legend_labels = cell(num_classes, 1);
TPR_all = cell(num_classes, 1);
FPR_all = cell(num_classes, 1);
AUC_values = zeros(num_classes, 1);

% Iterate over each class to compute TPR, FPR, and AUC
for i = 1:num_classes
    % Calculate scores for the current class
    scores_current = scores(:, i);
    true_labels = (y_test == i);

    % Sort the scores and true labels
    [~, idx] = sort(scores_current, 'descend');
    sorted_labels = true_labels(idx);

    % Calculate cumulative sums for true and false positives
    TP = cumsum(sorted_labels);
    FP = cumsum(~sorted_labels);

    % Normalize cumulative sums to get TPR and FPR
    TPR_bayesian = TP / sum(sorted_labels);
    FPR_bayesian = FP / sum(~sorted_labels);

    % Store TPR and FPR for later use in plotting
    TPR_all{i} = TPR_bayesian;
    FPR_all{i} = FPR_bayesian;

    % Calculate AUC using the trapezoidal rule
    AUC = trapz(FPR_bayesian, TPR_bayesian);
    AUC_values(i) = AUC;

    % Prepare legend label with AUC value
    legend_labels{i} = sprintf('Class %d (AUC = %.4f)', i, AUC);
end

% Plot the ROC curves
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

function loss = bayesObjective(params, X_train, y_train, cvpart)
    % Extract hyperparameters
    NumLearningCycles = params.NumLearningCycles;
    MinLeafSize = params.MinLeafSize;
    
    % Define the tree template with MinLeafSize
    t = templateTree('MinLeafSize', MinLeafSize);
    
    % Train model with given hyperparameters using the tree template
    mdl = fitcensemble(X_train, y_train, 'Method', 'Bag', ...
        'NumLearningCycles', NumLearningCycles, 'Learners', t);
    
    % Calculate cross-validated loss
    loss = kfoldLoss(crossval(mdl, 'CVPartition', cvpart));
end
