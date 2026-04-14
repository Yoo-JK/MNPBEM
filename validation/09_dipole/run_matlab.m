%% Dipole validation — MATLAB reference
%  20nm Au sphere, trisphere(144,20), z-dipole at [0,0,15]
%  Test 1: DipoleStat — tot+rad decay rate vs 500-700nm (21pt)
%  Test 2: DipoleRet  — same setup
%  Test 3: DipoleStat — decay rate vs z=12,15,20,30nm at lambda=520nm

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

data_dir = '/home/yoojk20/workspace/MNPBEM/validation/09_dipole/data';
fig_dir  = '/home/yoojk20/workspace/MNPBEM/validation/09_dipole/figures';

%% Common setup
epstab = {epsconst(1), epstable('gold.dat')};
op_s = bemoptions('sim', 'stat', 'interp', 'curv');
op_r = bemoptions('sim', 'ret',  'interp', 'curv');
p = comparticle(epstab, {trisphere(144, 20)}, [2, 1], 1, op_s);

enei = linspace(500, 700, 21);
n = length(enei);

%% ========== Test 1: DipoleStat — wavelength scan ==========
pt = compoint(p, [0, 0, 15]);
dip_s = dipole(pt, [0, 0, 1], op_s);
bem_s = bemsolver(p, op_s);

tot_s = zeros(1, n);
rad_s = zeros(1, n);

tic;
for i = 1:n
    sig = bem_s \ dip_s(p, enei(i));
    [t, r] = dip_s.decayrate(sig);
    tot_s(i) = t;
    rad_s(i) = r;
end
t_stat = toc;

T = table(enei', tot_s', rad_s', ...
    'VariableNames', {'wavelength_nm', 'tot', 'rad'});
writetable(T, fullfile(data_dir, 'matlab_dipole_stat.csv'));

figure('Visible', 'off', 'Position', [100, 100, 800, 500]);
plot(enei, tot_s, 'b-', 'LineWidth', 1.5); hold on;
plot(enei, rad_s, 'r--', 'LineWidth', 1.5);
xlabel('Wavelength (nm)');
ylabel('Decay rate');
title(sprintf('MATLAB DipoleStat — z-dipole at [0,0,15] (t=%.3fs)', t_stat));
legend('tot', 'rad', 'Location', 'best');
grid on;
saveas(gcf, fullfile(fig_dir, 'dipole_stat_matlab.png'));
close;

%% ========== Test 2: DipoleRet — wavelength scan ==========
p_r = comparticle(epstab, {trisphere(144, 20)}, [2, 1], 1, op_r);
pt_r = compoint(p_r, [0, 0, 15]);
dip_r = dipole(pt_r, [0, 0, 1], op_r);
bem_r = bemsolver(p_r, op_r);

tot_r = zeros(1, n);
rad_r = zeros(1, n);

tic;
for i = 1:n
    sig = bem_r \ dip_r(p_r, enei(i));
    [t, r] = dip_r.decayrate(sig);
    tot_r(i) = t;
    rad_r(i) = r;
end
t_ret = toc;

T2 = table(enei', tot_r', rad_r', ...
    'VariableNames', {'wavelength_nm', 'tot', 'rad'});
writetable(T2, fullfile(data_dir, 'matlab_dipole_ret.csv'));

figure('Visible', 'off', 'Position', [100, 100, 800, 500]);
plot(enei, tot_r, 'b-', 'LineWidth', 1.5); hold on;
plot(enei, rad_r, 'r--', 'LineWidth', 1.5);
xlabel('Wavelength (nm)');
ylabel('Decay rate');
title(sprintf('MATLAB DipoleRet — z-dipole at [0,0,15] (t=%.3fs)', t_ret));
legend('tot', 'rad', 'Location', 'best');
grid on;
saveas(gcf, fullfile(fig_dir, 'dipole_ret_matlab.png'));
close;

%% ========== Test 3: DipoleStat — distance dependence ==========
z_vals = [12, 15, 20, 30];
nz = length(z_vals);
lambda_fix = 520;

tot_dist = zeros(1, nz);
rad_dist = zeros(1, nz);

tic;
for j = 1:nz
    pt_d = compoint(p, [0, 0, z_vals(j)]);
    dip_d = dipole(pt_d, [0, 0, 1], op_s);
    sig = bem_s \ dip_d(p, lambda_fix);
    [t, r] = dip_d.decayrate(sig);
    tot_dist(j) = t;
    rad_dist(j) = r;
end
t_dist = toc;

T3 = table(z_vals', tot_dist', rad_dist', ...
    'VariableNames', {'z_nm', 'tot', 'rad'});
writetable(T3, fullfile(data_dir, 'matlab_dipole_distance.csv'));

figure('Visible', 'off', 'Position', [100, 100, 800, 500]);
plot(z_vals, tot_dist, 'bo-', 'LineWidth', 1.5); hold on;
plot(z_vals, rad_dist, 'rs--', 'LineWidth', 1.5);
xlabel('Dipole distance z (nm)');
ylabel('Decay rate');
title(sprintf('MATLAB DipoleStat distance — lambda=520nm (t=%.3fs)', t_dist));
legend('tot', 'rad', 'Location', 'best');
grid on;
saveas(gcf, fullfile(fig_dir, 'dipole_distance_matlab.png'));
close;

%% Timing CSV
T4 = table({'DipoleStat'; 'DipoleRet'; 'Distance'}, ...
    [t_stat; t_ret; t_dist], ...
    'VariableNames', {'test', 'time_sec'});
writetable(T4, fullfile(data_dir, 'matlab_timing.csv'));

fprintf('[info] MATLAB DipoleStat time: %.4f sec\n', t_stat);
fprintf('[info] MATLAB DipoleRet  time: %.4f sec\n', t_ret);
fprintf('[info] MATLAB Distance   time: %.4f sec\n', t_dist);
fprintf('[info] MATLAB validation complete.\n');
exit;
