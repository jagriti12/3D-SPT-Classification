function CD_mycheck_18jan
D = 0.1;
P = 5000;
T = 100;	% <#> Number of frames to simulate particles for	%
dt = 0.030;
Time = linspace(0, (T-1) * dt, T)';

traj = cell([P, 1]);  % Initialize Traj for this iteration
traj_tot = [];
dis_tot = [];
Bmin = 0.3;
Bmax = 0.3;

% Simulate motion for each trajectory with the chosen length
for p = 1:P
    traj{p} = ConfinedDiffusion(Time, D, Bmin, Bmax);
    traj_tot = [traj_tot; traj{p}];
    dis_tot = [dis_tot; traj{p}];
end
% dis_tot
MSD = [];
Time2 = Time(2:end);
ad = 1e-5;
for t = min(Time2):dt:max(Time2)
    dis_t = dis_tot(traj_tot(:,4)>=t-ad & traj_tot(:,4)<=t+ad,:);
    MSD_t = mean(sum(dis_t(:,1:3).^2,2));
 
    MSD = [MSD; MSD_t];
end
figure()
plot3(traj{1}(:,1),  traj{1}(:,2), traj{1}(:,3), '-b')
xlim([-5 5]); % Set x-axis limits
ylim([-5 5]); % Set y-axis limits
zlim([-5 5]); % Set z-axis limits

xlabel('X (µm)'); % Label for x-axis
ylabel('Y (µm)'); % Label for y-axis
zlabel('Z (µm)'); % Label for z-axis

rotate3d on;
figure()
plot(Time2, MSD, 'ob', ...
    Time2, 2*3*D.*Time2, '--k')
legend('Simulation', 'Theory: MSD = 6Dt','Location','northwest')
xlabel('Time, sec')
ylabel('MSD')
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
    
%     disp(['Randomly selected B_param: ', num2str(B_param)]); % Display the selected B_param
    % Calculate the radius of confinement from B_param
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
function [traj] = NormalDiffusion(t, D)
N = length(t);	% N - Number of positions, N-1 - number of displacements 				%
dt = diff(t);	% Time differential between positions			%
	
traj = zeros([N-1, 3+1]);	% Initialize the trajectory %
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

