function ND_mycheck_18jan
D = 3;
T = 100;	% <#> Number of frames to simulate particles for	%
P = 500;	% <#> Number of particles to simulate				%
dt = 0.030;
Time = (linspace(0, (T-1) * dt, T))';


traj = cell([P, 1]);  % Initialize Traj for this iteration
traj_tot = [];
dis_tot = [];
for p = 1:P
    traj{p} = NormalDiffusion(Time, D);
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
legend('Estimation', 'Theory: MSD = 6Dt','Location','northwest')
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
figure()
plot(Time2, MSD./(2*3*D.*Time2), 'ob', Time2, ones(size(Time2)), '--k')
xlabel('Time, sec')
ylabel('MSD_{est} / MSD_{teor}')
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