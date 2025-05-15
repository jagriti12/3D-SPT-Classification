function DM_mycheck_18jan
D = 2;
P = 500;
T = 100;	% <#> Number of frames to simulate particles for	%
dt = 0.030;
Time = (linspace(0, (T-1) * dt, T))';
kdm1 = 5; kdm2 = 100;
    
traj_tot = [];
dis_tot = [];
for p = 1:P
    traj{p} = DirectedMotion(Time, D, kdm1, kdm2);
    traj_tot = [traj_tot; traj{p}];
    dis_tot = [dis_tot; traj{p}];
end
    
% dis_tot
MSD = [];
ad = 1e-5;
Time2 = Time(2:end);
for t = min(Time2):dt:max(Time2)
    dis_t = dis_tot(traj_tot(:,4)>=t-ad & traj_tot(:,4)<=t+ad,:);
    MSD_t = mean(sum(dis_t(:,1:3).^2,2));
 
    MSD = [MSD; MSD_t];
end



figure()
plot(Time2, MSD, 'or', ...
    Time2, 2*3*D.*Time2, 'k--')
legend('Simulation', 'Theory: MSD = 6Dt','Location','northwest')
xlabel('Time, sec')
ylabel('MSD')

figure()
plot3(traj{1}(:,1),  traj{1}(:,2), traj{1}(:,3), '-b')
xlim([-5 5]); % Set x-axis limits
ylim([-5 5]); % Set y-axis limits
zlim([-5 5]); % Set z-axis limits

xlabel('X (µm)'); % Label for x-axis
ylabel('Y (µm)'); % Label for y-axis
zlabel('Z (µm)'); % Label for z-axis
rotate3d on; % Enable interactive rotation
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
traj = NormalDiffusion(t, D);	% Initialize the trajectory with diffusion %

phi = atan2(v(2), v(1));
theta = acos(v(3)/sqrt(sum(v.^2)));
    
omega_phi = Wiener(t, 0, D/100);%
omega_theta = Wiener(t, 0, D/100);%
phi = phi + cumsum(omega_phi .* dt);
theta = theta + cumsum(omega_theta .* dt);
	
vel = speed * [cos(phi).*sin(theta) sin(phi).*sin(theta) cos(theta)];

traj(:,1:3) = traj(:,1:3) + cumsum(vel .* dt, 1);
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
















