function AD_mycheck_18jan
D = 2;	% <um^2/s> Diffusion Coefficien
T = 100;	% <#> Number of frames to simulate particles for	%
P = 10;	% <#> Number of particles to simulate				%
dt = 0.03;

Time = linspace(0, (T-1) * dt, T)';

traj = cell([P, 1]);  % Initialize Traj for this iteration
traj_tot = [];
dis_tot = [];

amin = 0.7;
amax = 0.7;
 
% Simulate motion for each trajectory with the chosen length
for p = 1:P
    traj{p} = AnomalousDiffusion(Time, D, amin, amax);
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
plot3(traj{1}(:,1),  traj{1}(:,2), traj{1}(:,3), '-b')
xlim([-5 5]); % Set x-axis limits
ylim([-5 5]); % Set y-axis limits
zlim([-5 5]); % Set z-axis limits

xlabel('X (µm)'); % Label for x-axis
ylabel('Y (µm)'); % Label for y-axis
zlabel('Z (µm)'); % Label for z-axis

figure()
plot(Time2, MSD, 'ob', ...
    Time2, 2*3*D.*Time2, '--k')
legend('Simulatuon', 'Theory: MSD = 6Dt', 'Location','northwest')
xlabel('Time, sec')
ylabel('MSD')
assignin('base', 'traj', traj);
assignin('base', 'traj_tot', traj_tot);
end


function [traj] = AnomalousDiffusion(t, D, alphamin, alphamax)
	% For sub-diffusive motion, Wagner, et al. uses the Weierstrass-Mandelbrot fxn
	% The sum is taken from n = -8 to +48 as described by Saxton
	
	%% Initialize %%
	T = length(t);	% T - Number of positions, T-1 - number of displacements 				%
	dt = diff(t);	% Time differential between positions			%
    alpha = alphamin + (alphamax - alphamin) * rand;
	
	traj = zeros([T-1, 3 + 1]);	% Initialize the trajectory %
	traj(:,end) = t(2:end,:);
	
	n = -8:48;		% As described by Saxton %
	gamma = sqrt(pi);
	t_ = 2*pi/max(t) * t(2:end,:);
	
	phi = 2*pi * rand([3, length(n)]);	% Random phase %
	
	%% Evaluate %%
	% Determine the trajectory from the Weierstrass-Mendelbrot function %
	for d = 1:3
		% Substitutions for quality of life %
		num = cos(phi(d,:)) - cos(t_ * gamma.^n + phi(d,:));
 		den = gamma.^(- alpha * n / 2);

		W = sum( num .* den, 2);	% W(t) %
		
		% Append to the trajectory %
		traj(:,d) = W;
	end
	
	% Rescale such that <r_1^2> = 6 D dt %
	sqdisp = mean(sum((traj(2:end,1:3) - traj(1:end-1,1:3)).^2, 2));
	traj(:,1:3) = traj(:,1:3) * sqrt(6*D*mean(dt)/sqdisp);
	%% Output %%
	% traj %
end
function [A, alpha] = const_AD_calc(D, MSD, t)
par0 = [1 1];
options = optimset('Display', 'iter-detailed');
[par, resnorm,residuals,~,~,~,jacobian] = lsqnonlin(@(v) OF(v, D, MSD, t), par0, [], [], options);
 ci = nlparci(par,residuals,'jacobian',jacobian);
 m  = ([par' (ci(:,2) - ci(:,1)) / 2./par'*100])';
 fprintf('%25s %25s\n', 'X', 'DX,%');
 fprintf('%25.8f %25.8f\n', m);
 fprintf(' resnorm=%.6f\n',resnorm);
 fprintf(' %.8f', par);
 fprintf('\n');


A = par(1);
alpha = par(2);

end
function d = OF(par, D, MSD_sim, t)
MSD_calc = MSD_fit_calc(par(1), par(2), D, t);
d = MSD_sim(2:end,:) - MSD_calc(2:end,:);
end
function MSD = MSD_fit_calc(A, alpha, D, t)
% MSD = rc.^2.*(1 - A.*exp(-B.*t./rc.^2));
MSD = A.*6.*D.*t.^alpha;
end