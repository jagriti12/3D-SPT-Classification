%% Code Updated by Jagriti Chatterjee %%
% Notes on parameters %
% MinLen: 10-30, no change in any performance metric. Go with lowest value
% WinLen: 11-41, Plateaus the higher you go, 15, 17, 23, 35 seem good numbers
% Trees:  10-100, Plateaus after 30, 55 might be a peak.
% T: 10-200, Only really important above the Window Length. Selecting 100.
% P: 100-2000, Also really doesn't matter much, probably around 300 should work?
% Layers: 1-10, Slight improvements after 3, peaks at 7, but at the cost of overfit? (20% data trained)

%% --- GLOBAL DECLARATIONS --- %%
rng(0);
tic
% Redefine the `clrs` array just before it's used
clrs = [
    0.929, 0.635, 0.466;  % Custom color
    0.500, 0.500, 0.500;  % Grey
    0.300, 0.700, 0.900;  % Light blue
    0.800, 0.100, 0.100;  % Reddish
];

% disp('Current colors array:');
% disp(clrs);

%% --- USER PARAMETERS --- %%
% Simulation Parameters %
T = 100;    % <#> Number of frames to simulate particles for    %
P = 5000;    % <#> Number of particles to simulate                %

% Diffusion Coefficient Range
D_min = 0.01;  % <um^2/s> Minimum Diffusion Coefficient
D_max = 15;    % <um^2/s> Maximum Diffusion Coefficient

Mixed = true;    % <bool> Simulate mixed motion or consistent motion?        %
MinLen = 20;    % <#> Minimum length for a segment of one kind of dynamics    %
WinLen = 29;    % <#> Window length for analysis %

Trees = 100;        % <#> Number of trees to use when training the random forest    %
Layers = 10;        % <#> Number of layers of splits allowed in the random forest    %

%% --- INITIALIZE --- %%
Traj = cell([P, 1]);    % [x,y,z,t] Position vector through time    %
Lbls = cell([P, 1]);    % <1,2,3> Dynamics labels                    %

Wtrj = cell([P, T]);
Wlbl = nan([P, T]);

Time = linspace(0, (T-1) * 0.030, T)';

%%
Range = [1];
F1 = zeros([length(Range), 4]);
TPR = zeros([length(Range), 4]);
TNR = zeros([length(Range), 4]);
Acc = zeros([length(Range), 1]);

%% --- SIMULATION LOOP --- %%
for val = 1:length(Range)
    %% --- SIMULATE MOTION --- %%
    Traj = cell([P, 1]);  % Initialize Traj for this iteration
    Lbls = cell([P, 1]);  % Initialize Lbls for this iteration
    
    % Simulate motion for each trajectory with the random length of segments
    for p = 1:P
        % Sample D within the specified range
        D = D_min + (D_max - D_min) * rand;
        % Simulate motion with the chosen MinLen
        [Traj{p}, Lbls{p}] = SimulateMixedMotion(Time, D, MinLen);
    end

    pick = 20;    % Example trajectory to pick for visualization
    
    % Ensure Traj{pick} has at least 2 columns
    if size(Traj{pick}, 2) < 2
        error('Trajectory %d does not have at least 2 dimensions.', pick);
    end
    
    % Ensure Traj{pick} has 3 columns by adding a zero third dimension if necessary
    if size(Traj{pick}, 2) == 2
        warning('Trajectory %d is 2D. Proceeding with 2D analysis.', pick);
        Traj{pick} = [Traj{pick}, zeros(size(Traj{pick}, 1), 1)];
    end
    
    % Now Traj{pick} should have 3 columns
    if size(Traj{pick}, 2) < 3
        error('Trajectory %d does not have enough dimensions for 3D analysis.', pick);
    end
    
    Fmax = max(Traj{pick}, [], 1) / 0.030;
    Fmin = min(Traj{pick}, [], 1) / 0.030;

    % Ensure Fmax and Fmin have at least 3 elements
    if numel(Fmax) < 3 || numel(Fmin) < 3
        error('Trajectory %d does not have enough dimensions for 3D analysis.', pick);
    end
    
    % Calculate the length of the segment in the third dimension
    Len = Fmax(3) - Fmin(3) + 1;


    % Plot the trajectory
    figure(1);
    plt = plot3(Traj{pick}(:,1),Traj{pick}(:,2), Traj{pick}(:,3), 'linewidth', 2);
    drawnow;
    
    grid on;
    axis equal;

    xlim([min(Traj{pick}(:,1))-0.5, max(Traj{pick}(:,1))+0.5]);
    ylim([min(Traj{pick}(:,2))-0.5, max(Traj{pick}(:,2))+0.5]);
    zlim([min(Traj{pick}(:,3))-0.5, max(Traj{pick}(:,3))+0.5]);
    xticklabels([]);
    yticklabels([]);
    zticklabels([]);
    view(30, 30);
    
    [sub_dm, sub_nd, sub_ad, sub_co] = Subtrajectories({Traj{pick}}, {Lbls{pick}});
    
    figure(2); clf(); hold on;
    for p = 1:length(sub_dm)
        plt = plot3(sub_dm{p}(:,1), sub_dm{p}(:,2), sub_dm{p}(:,3), 'color', 'cyan', 'linewidth', 2);
        drawnow;
    end
    for p = 1:length(sub_nd)
        plt = plot3(sub_nd{p}(:,1), sub_nd{p}(:,2), sub_nd{p}(:,3), 'color', 'black', 'linewidth', 2);
        drawnow;
    end
    for p = 1:length(sub_ad)
        plt = plot3(sub_ad{p}(:,1), sub_ad{p}(:,2), sub_ad{p}(:,3), 'color', 'green', 'linewidth', 2);
        drawnow;
    end
    for p = 1:length(sub_co)
        plt = plot3(sub_co{p}(:,1), sub_co{p}(:,2), sub_co{p}(:,3), 'color', 'red', 'linewidth', 2);
        drawnow;
    end

    grid on;
    axis equal;
    xlim([min(Traj{pick}(:,1))-0.2, max(Traj{pick}(:,1))+0.2]);
    ylim([min(Traj{pick}(:,2))-0.2, max(Traj{pick}(:,2))+0.2]);
    zlim([min(Traj{pick}(:,3))-0.2, max(Traj{pick}(:,3))+0.2]);
    xticklabels([]);
    yticklabels([]);
    zticklabels([]);
    view(30, 30);
    
    %% --- WINDOW TRAJECTORIES --- %%
    wb = waitbar(0, 'Windowing motion...');
    
    Wtrj = cell([P, T - 2*round((WinLen-1)/2)]);
    Wlbl = nan([P, T - 2*round((WinLen-1)/2)]);

    for p = 1:P
            F = size(Traj{p}, 1);
            NN = 1;
            for f = 2:F % instead of f = 1:F
                % Establish the moving window %
                lft = max(f - floor(WinLen/2), 1);
                rgt = min(f + floor(WinLen/2), F);
                if (rgt - lft + 1) == WinLen %Updated part
                    % Snip the trajectory at this point %
                    Wtrj{p,NN} = Traj{p}(lft:rgt, :); %instead of Wtrj{p,f} = Traj{p}(lft:rgt, :);
                    if(Mixed),    Wlbl(p,NN) = Lbls{p}(f); %instead of Wlbl(p,f) = Lbls{p}(f);
                    else,        Wlbl(p,NN) = Lbls{p}; %instead of Wlbl(p,f) = Lbls{p};
                    end
                    Wtrj{p,NN} = Wtrj{p,NN} - ones(size(Wtrj{p,NN},1),1) * Wtrj{p,NN}(1,:); % each Wtrj matrix starts with [0 0 0 0]
                    NN = NN + 1;
                end
            end
         
            % Update the user %
            if(mod(p,round(sqrt(P))) == 0)
                waitbar(p/P, wb, sprintf('Windowing motion... %5.2f%%', p/P*100));
            end
        end
 
    close(wb);
    
    % Ensure Wlbl only contains labels 1 to 4
    %Wlbl = arrayfun(@(x) max(1, min(4, x)), Wlbl);



    %% --- FEATURE EXTRACTION --- %%
    ftrs = RandForest_Features_all(Wtrj(:));

    figure(99); clf();
    strs = {'Alpha', 'AngularGaussianityIndex','Asymmetry','Avg. MSD Ratio', 'Efficiency','Fractal Dimension','Gaussianity',  'JumpLength','Kurtosis',  'MaximalExcursion','MeanMaximalExcursion', 'Straightness', 'Trappedness', 'VelocityAutocorrelation'};
    clrs = colormap('lines');

    for k = 1:14  % Assuming 14 features for example
        mesh = linspace(min(ftrs(k,:)), max(ftrs(k,:)), 400);
        subplot(7, 2, k); hold on;
        
        for cat = 1:4  % Assuming there are 4 categories
            f_cat = ksdensity(ftrs(k, Wlbl == cat), mesh);
            colorIndex = mod(cat-1, size(clrs, 1)) + 1;
            
            % % Extract the RGB triplet just before plotting
            currentColor = clrs(colorIndex, :);
            % disp(['Color Index: ', num2str(colorIndex), ' RGB Triplet: ', mat2str(currentColor)]);  % Debug output

            if numel(currentColor) ~= 3
                error('Invalid RGB triplet at index %d: [%s]', colorIndex, mat2str(currentColor));
            end

            plot(mesh, f_cat / max(f_cat), 'linewidth', 2, 'color', currentColor);
            line(mean(ftrs(k, Wlbl == cat)) * [1,1], [0, 1], 'color', currentColor, 'linestyle', '--');
        end
        
        title(strs{k});
        ylabel('Normalized Occurrence');
    end

    % disp(['Using color index: ', num2str(colorIndex)]);
    % disp(['RGB Triplet: ', mat2str(clrs(colorIndex, :))]);

    drawnow;

    tic
    %% --- TRAINING --- %%
    % Get training data %
    P_train = floor(size(ftrs,2) * 0.90);
    idx = randperm(size(ftrs,2), P_train);

    train_ftrs = ftrs(:, idx);
    train_lbls = Wlbl(idx);
    % Assuming `train_lbls` is a column vector
    train_lbls(train_lbls == 0) = 1;
    %% Combine Trajectory training features with labels %%
    combined_train_ftrs_lbls = vertcat(train_ftrs, train_lbls);
    combined_train_ftrs_lbls = transpose(combined_train_ftrs_lbls);

    % Train the random forest %
    fprintf('Training random forest with %i trees...\t', Trees);
    mdl_rt = TreeBagger(Trees, train_ftrs', train_lbls, 'OOBPrediction', 'on', 'maxnumsplits', 2^Layers - 1);
    fprintf('Done!\n');

    %% Plot OOB Error Rate
    oobErrorRates = oobError(mdl_rt);  % Correctly extract OOB error rates
    numTrees = 1:mdl_rt.NumTrees; % Gets the correct number of trees from the model

    figure;
    plot(numTrees, oobErrorRates, 'b-', 'LineWidth', 2);
    xlabel('Number of Trees');
    ylabel('OOB Error Rate');
    title('Random Forest OOB Error Rate Across Number of Trees');
    grid on;

    %% --- TESTING --- %%
    % Get testing data - the remaining data not used in training %
    idx_ = [];
    for i = 1:size(ftrs, 2)
        if(~any(idx == i)), idx_(end+1) = i; end
    end
    test_ftrs = ftrs(:, idx_);
    test_lbls = Wlbl(idx_);
    test_lbls(test_lbls == 0) = 1;
    %% Combine Trajectory training features with labels %%
    combined_test_ftrs_lbls = vertcat(test_ftrs, test_lbls);
    combined_test_ftrs_lbls = transpose(combined_test_ftrs_lbls);
    % Predict %
    test_pred = predict(mdl_rt, test_ftrs');
    for i = 1:length(test_pred), test_pred{i} = str2double(test_pred{i}); end
    test_pred = [test_pred{:}];

    % Evaluation %
    conf = zeros([4, 4]);
    for i = 1:4
        for j = 1:4
            conf(i,j) = sum( (test_pred == i) & (test_lbls == j) );
        end
    end

    for i = 1:4
        F1(val,i) = 2*conf(i,i) / (sum(conf(i,:)) + sum(conf(:,i)));
        TPR(val,i) = conf(i,i) / sum(conf(:,i));
        TNR(val,i) = conf(i,i) / sum(conf(i,:));
    end
    Acc(val) = sum(diag(conf))/sum(conf(:));

    fprintf('--- Testing Results #%i/%i ---\n\n', val, length(Range));
    disp(conf)
    disp([F1(val,:); TPR(val,:); TNR(val,:)]);
    disp(Acc(val));
end
toc

%% Visualize Performance Metrics %%
figure(1);
subplot(2,2,1);
plot(Range, Acc, '.-');
title('Accuracy');
ylim([-inf, 1]);

subplot(2,2,3);
plot(Range, F1, '.-');
title('F1 Score');
ylim([-inf, 1]);

subplot(2,2,2);
plot(Range, TPR, '.-');
title('Sensitivity');
ylim([-inf, 1]);

subplot(2,2,4);
plot(Range, TNR, '.-');
title('Specificity');
ylim([-inf, 1]);

[~, scores] = predict(mdl_rt, test_ftrs');

% Get the number of classes
num_classes = size(scores, 2);

% Define the colors for each class
colors = {'r', 'g', 'b', 'm'};

% Create a figure to plot the ROC curves
figure;

% Create a cell array to store the legend labels
legend_labels = cell(num_classes, 1);

% Iterate over each class
for i = 1:num_classes
    % Sort the scores and true labels for the current class
    [~, idx] = sort(scores(:, i), 'descend');
    sorted_labels = test_lbls(idx);

    % Calculate the true positive rate (TPR) and false positive rate (FPR)
    TP = cumsum(sorted_labels == i);
    FP = cumsum(sorted_labels ~= i);
    TPR = TP / sum(sorted_labels == i);
    FPR = FP / sum(sorted_labels ~= i);

    % Calculate the AUC using the trapezoidal rule approximation
    AUC = trapz(FPR, TPR);
    
    % Store the legend label with AUC value
    legend_labels{i} = sprintf('Class %d (AUC = %.4f)', i, AUC);

    % Assign a color to the current class
    color = colors{mod(i-1, length(colors))+1};

    % Plot the ROC curve for the current class with the assigned color
    plot(FPR, TPR, 'Color', color, 'LineWidth', 2);
    hold on;
end

hold on;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves for Each Class');
legend(legend_labels);
grid on;

toc

%% --- TRAJECTORY SIMULATIONS --- %%
% There are four types of motion defined in Wagner, et al.:
%    (ND) Normal Diffusion        <r^2(n)> = 4 D n dt
%    (AD) Anomalous Diffusion     <r^2(n)> = 4 D (n dt)^alpha
%    (DM) Directed Motion        <r^2(n)> = 4 D n dt + (v n dt)^2
%    (CD) Confined Diffusion        <r^2(n)> ~ R^2 [1 - A exp(-4 B D n dt / R^2)]
%
% Where
%    D = Diffusion Coefficient
%    alpha = Exponent ( < 1 )
%    v = Velocity
%    R = Radius of Confinement
%    A, B = Characterize the shape of confinement
%
% The appropriate functions to generate each one of these types of motion are 
% detailed in the following functions.

function [trj, lbl] = SimulateConsistentMotion(t, D)
    %% Initialize %%
    lbl = randi(4);
    alpha_min = 0.1; alpha_max = 0.7;   % AD parameters
    Bmin = 0.2;   Bmax = 1.2;               % CD parameters
    kdm1 = 5; kdm2 = 100;               % DM parameters

    %% Evaluate %%
    switch(lbl)
        case 1
            trj = DirectedMotion(t, D, kdm1, kdm2);
        case 2
            trj = NormalDiffusion(t, D);
        case 3
            trj = AnomalousDiffusion(t, D, alpha_min, alpha_max);
        case 4
            trj = ConfinedDiffusion(t, D, Bmin, Bmax);
    end
end

function [trjs, lbls] = SimulateMixedMotion(t, D, minLen)
    F = length(t);
    trjs = zeros(F, 4); % Assuming 4 columns: x, y, z, and time (or another variable)
    lbls = zeros(F, 1);
    seg_firstpoint = [0 0 0 0];
    f = 1;
    while(f < F)
        seg_len = randi([minLen, F - f]);
        if (F - f - seg_len) >= minLen
            df = seg_len;
        else
            df = F - f;
        end
        t_ = t(f : f + df);
        [seg, lbl] = SimulateConsistentMotion(t_, D);

        % If seg is missing the fourth column, add it
        if size(seg, 2) == 3
            seg = [seg, t_(1:size(seg, 1))]; % Append the time or the missing variable
        end

        % Check size before assignment to avoid dimension mismatch
        if size(seg, 2) ~= size(trjs, 2)
            error('Dimension mismatch. Expected size: %d, Got: %d', size(trjs, 2), size(seg, 2));
        end
        mat_seg_firstpoint = ones(size(seg,1),1) * seg_firstpoint;
        trjs(f + 1 : f + df, :) = seg + mat_seg_firstpoint;
        lbls(f + 1 : f + df) = lbl;
        seg_firstpoint(1:3) = trjs(f + df, 1:3);
        f = f + df;
    end

    % Drop the first row
    % trjs = trjs(2:end, :);
    % lbls = lbls(2:end);
end


function [traj] = NormalDiffusion(t, D)
    N = length(t);    % N - Number of positions, N-1 - number of displacements                 %
    dt = diff(t);    % Time differential between positions            %
        
    traj = zeros([N-1, 3+1]);    % Initialize the trajectory %
    traj(:,end) = t(2:end);

    % Normal distribution
    mu = 0;
    sigma = (2*D*dt).^0.5;
    du_x = mu + sigma .* randn(N-1, 1);
    du_y = mu + sigma .* randn(N-1, 1);
    du_z = mu + sigma .* randn(N-1, 1);
    du_3d = [du_x du_y du_z];
    traj(:,1:3) = cumsum(du_3d);
end

function [traj] = AnomalousDiffusion(t, D, alphamin, alphamax)
    % For sub-diffusive motion, Wagner, et al. uses the Weierstrass-Mandelbrot fxn
    % The sum is taken from n = -8 to +48 as described by Saxton
    
    %% Initialize %%
    T = length(t);    % T - Number of positions, T-1 - number of displacements                 %
    dt = diff(t);    % Time differential between positions            %
    alpha = alphamin + (alphamax - alphamin) * rand;
    
    traj = zeros([T-1, 3 + 1]);    % Initialize the trajectory %
    traj(:,end) = t(2:end,:);
    
    n = -8:48;        % As described by Saxton %
    gamma = sqrt(pi);
    t_ = 2*pi/max(t) * t(2:end,:);
    
    phi = 2*pi * rand([3, length(n)]);    % Random phase %
    
    %% Evaluate %%
    % Determine the trajectory from the Weierstrass-Mendelbrot function %
    for d = 1:3
        % Substitutions for quality of life %
        num = cos(phi(d,:)) - cos(t_ * gamma.^n + phi(d,:));
        den = gamma.^(- alpha * n / 2);

        W = sum( num .* den, 2);    % W(t) %
        
        % Append to the trajectory %
        traj(:,d) = W;
    end
    
    % Rescale such that <r_1^2> = 6 D dt %
    sqdisp = mean(sum((traj(2:end,1:3) - traj(1:end-1,1:3)).^2, 2));
    traj(:,1:3) = traj(:,1:3) * sqrt(6*D*mean(dt)/sqdisp);
    %% Output %%
    % traj %
end

function [traj] = DirectedMotion(t, D, kdm1, kdm2)
    % Random DM coefficient calculation: lbd - kdm, ubd - 1000
    coef = kdm1 + (kdm2 - kdm1) * rand;
        
    % Speed module calculation 
    speed = coef * sqrt(D);
        
    dt = diff(t);
    % Simulate motion for each trajectory with the chosen length
    vel = randn([3, 1]);    
    v = vel ./ norm(vel);
    traj = NormalDiffusion(t, D);    % Initialize the trajectory with diffusion %

    phi = atan2(v(2), v(1));
    theta = acos(v(3)/sqrt(sum(v.^2)));
        
    omega_phi = Wiener(t, 0, D/100);%
    omega_theta = Wiener(t, 0, D/100);%
    phi = phi + cumsum(omega_phi .* dt);
    theta = theta + cumsum(omega_theta .* dt);
        
    vel = speed * [cos(phi).*sin(theta) sin(phi).*sin(theta) cos(theta)];

    traj(:,1:3) = traj(:,1:3) + cumsum(vel .* dt, 1);
end

function [traj, B] = ConfinedDiffusion(t, D, Bmin, Bmax)
    % Inputs:
    %   t - Time vector
    %   D - Diffusion coefficient
    %   B_param - Dimensionless parameter for confinement
    % Output:
    %   traj - Simulated 3D trajectory of the diffusion process
    %   B - Dimensionless parameter characterizing confinement

    N = length(t);  % Number of time points
    dt = mean(diff(t));  % time step for steps
    ddt = mean(diff(t)) / 100;  % Smaller time step for sub-steps
    t_mini = (0:ddt:dt)';
    % Randomly choose B between Bmin and Bmax
    B_param = Bmin + rand() * (Bmax - Bmin);
    
    % Calculate the radius of confinement from B_param
    % r = sqrt(D * (N-1) * dt) / B_param.^(1/3);
    r = sqrt(D) / B_param.^(1/3);
   
    % Initialize the trajectory
    traj = zeros(N-1, 4);  % Start at the center
    traj_mini_res = zeros(N-1, 3);

    % Main simulation loop
    k = 1;
    while k <= N-1
        % Sub-steps
        traj_mini = NormalDiffusion(t_mini, D);
        traj_mini_res(k, :) = traj_mini(end,1:3);
        traj_test = cumsum(traj_mini_res);
        len = sqrt(traj_test(end,1).^2 + traj_test(end,2).^2 + traj_test(end,3).^2);
        if len <= r
            k = k + 1; % Accept the step
        end
    end
    traj(:,1:3) = cumsum(traj_mini_res);

    % Calculate the radius of gyration as an approximation for the ellipsoid fitting
    traj_center = mean(traj, 1);
    rg = sqrt(mean(sum((traj - traj_center).^2, 2)));  % Radius of gyration

    % Approximate the volume of the smallest ellipsoid
    V_ell = (4/3) * pi * rg^3;

    % Define the ratio B
    B = V_ell ./ ((4/3) * pi * r.^3);
    
    traj(:,4) = t(2:end);
end

%% --- HELPER FUNCTIONS --- %%
% Additional functions to improve readability %

function [res] = Wiener(time, drift, variance)
    % Simulates a Wiener process, nondifferentiable random motion %
    N = length(time);
    X = randn(N-1, 1); % Ensure X is a single-column vector
    dt = diff(time(:)); % Ensure dt is a single-column vector
    vel = drift .* dt + variance .* X .* sqrt(dt);
    res = cumsum(vel);

    % Check if the result is a single-column vector
    if size(res, 2) ~= 1
        error('Wiener process output is not a single-column vector.');
    end
end

function [sub_dir, sub_dif, sub_anom, sub_con] = Subtrajectories(traj, class)
    % Split up the trajectories in traj based on contiguous classification %
    P = length(traj);
    sub_dir = {};
    sub_dif = {};
    sub_anom = {};
    sub_con = {};
    
    for p = 1:P
        % Run through this trajectory and break it apart into segments of
        % similar classifications
        cval = class{p}(1);
        subtraj = traj{p}(1,:);
        
        for f = 2:length(class{p})
            if((class{p}(f) == cval) && (f < length(class{p})))
                % Append this frame to the current subtrajectory %
                subtraj(end+1,:) = traj{p}(f,:);
            elseif(size(subtraj, 1) > 1)
                % Save the current subtrajectory %
                subtraj(end+1,:) = traj{p}(f,:);
                if(cval == 1),        sub_dir{end+1} = subtraj;
                elseif(cval == 2),    sub_dif{end+1} = subtraj;
                elseif(cval == 3),    sub_anom{end+1} = subtraj;
                else,                sub_con{end+1} = subtraj;
                end
                
                % Add the current point to a new subtrajectory %
                subtraj = [];
                subtraj(1,:) = traj{p}(f,:);
                cval = class{p}(f);
            else
                % No sense in having single points in there %
                cval = class{p}(f);
            end
        end
    end
end
