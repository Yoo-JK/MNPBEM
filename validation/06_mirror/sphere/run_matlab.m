%% 06_mirror / sphere — MATLAB
%  BEMStatMirror / BEMRetMirror vs BEMStat / BEMRet
%  1/4 sphere (trispheresegment) + sym='xy'
%  Gold sphere, d=20nm, 400-800nm (41pt)

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

script_dir = fileparts(mfilename('fullpath'));
data_dir   = fullfile(script_dir, 'data');
fig_dir    = fullfile(script_dir, 'figures');
if ~exist(data_dir, 'dir'); mkdir(data_dir); end
if ~exist(fig_dir,  'dir'); mkdir(fig_dir);  end

epstab = {epsconst(1), epstable('gold.dat')};
op     = bemoptions('sim','stat','interp','curv');
op_ret = bemoptions('sim','ret','interp','curv');
enei = linspace(400, 800, 41);

n = 13;
phi   = linspace(0, pi/2, n);
theta = linspace(0, pi, 2*n - 1);
p_seg = trispheresegment(phi, theta, 20);

%% Quasistatic Mirror
p_mir = comparticlemirror(epstab, {p_seg}, [2, 1], 1, op, 'sym', 'xy');
bem_m = bemstatmirror(p_mir, op);
exc_m = planewavestatmirror([1,0,0; 0,1,0; 0,0,1]);

ext_sm = zeros(length(enei), 3);
tic;
for i = 1:length(enei)
    sig = bem_m \ exc_m(p_mir, enei(i));
    ext_sm(i, :) = extinction(exc_m, sig);
end
t_sm = toc;

%% Quasistatic Full (same expanded mesh)
p_full = full(p_mir);
bem_f  = bemstat(p_full, op);
exc_f  = planewavestat([1,0,0; 0,1,0; 0,0,1]);

ext_sf = zeros(length(enei), 3);
tic;
for i = 1:length(enei)
    sig = bem_f \ exc_f(p_full, enei(i));
    ext_sf(i, :) = extinction(exc_f, sig);
end
t_sf = toc;
fprintf('[info] stat mirror %.3fs, full %.3fs, speedup %.2fx\n', t_sm, t_sf, t_sf/t_sm);

%% Retarded Mirror (bemsolver + planewave auto-dispatch)
op_ret_sym = bemoptions('sim','ret','interp','curv','sym','xy');
p_mir_r = comparticlemirror(epstab, {p_seg}, [2, 1], 1, op_ret_sym);
bem_mr = bemsolver(p_mir_r, op_ret_sym);
exc_mr = planewave([1,0,0; 0,1,0], [0,0,1; 0,0,1], op_ret_sym);

ext_rm = zeros(length(enei), 2);
tic;
for i = 1:length(enei)
    sig = bem_mr \ exc_mr(p_mir_r, enei(i));
    ext_rm(i, :) = exc_mr.sca(sig);
end
t_rm = toc;

%% Retarded Full
p_full_r = full(p_mir_r);
bem_fr   = bemret(p_full_r, op_ret);
exc_fr   = planewave([1,0,0; 0,1,0], [0,0,1; 0,0,1], op_ret);

ext_rf = zeros(length(enei), 2);
tic;
for i = 1:length(enei)
    sig = bem_fr \ exc_fr(p_full_r, enei(i));
    ext_rf(i, :) = exc_fr.sca(sig);
end
t_rf = toc;
fprintf('[info] ret  mirror %.3fs, full %.3fs, speedup %.2fx\n', t_rm, t_rf, t_rf/t_rm);

%% Save CSVs
T = table(enei', ext_sm(:,1), ext_sm(:,2), ext_sm(:,3), ...
    'VariableNames', {'wavelength_nm','ext_x','ext_y','ext_z'});
writetable(T, fullfile(data_dir,'stat_mirror_matlab.csv'));

T = table(enei', ext_sf(:,1), ext_sf(:,2), ext_sf(:,3), ...
    'VariableNames', {'wavelength_nm','ext_x','ext_y','ext_z'});
writetable(T, fullfile(data_dir,'stat_full_matlab.csv'));

T = table(enei', ext_rm(:,1), ext_rm(:,2), ...
    'VariableNames', {'wavelength_nm','ext_x','ext_y'});
writetable(T, fullfile(data_dir,'ret_mirror_matlab.csv'));

T = table(enei', ext_rf(:,1), ext_rf(:,2), ...
    'VariableNames', {'wavelength_nm','ext_x','ext_y'});
writetable(T, fullfile(data_dir,'ret_full_matlab.csv'));

T = table({'stat_mirror';'stat_full';'ret_mirror';'ret_full'}, [t_sm; t_sf; t_rm; t_rf], ...
    'VariableNames', {'case','time_sec'});
writetable(T, fullfile(data_dir,'matlab_timing.csv'));

%% Plots
fig = figure('Visible','off','Position',[100 100 800 500]);
plot(enei, ext_sm(:,1), 'b-', 'LineWidth',1.5); hold on;
plot(enei, ext_sm(:,2), 'r--','LineWidth',1.5);
plot(enei, ext_sm(:,3), 'g:', 'LineWidth',1.5);
xlabel('Wavelength (nm)'); ylabel('Extinction (nm^2)');
title(sprintf('MATLAB stat mirror sphere (t=%.3fs)', t_sm));
legend('x','y','z','Location','best'); grid on;
saveas(fig, fullfile(fig_dir,'stat_mirror_matlab.png')); close(fig);

fig = figure('Visible','off','Position',[100 100 800 500]);
plot(enei, ext_rm(:,1), 'b-', 'LineWidth',1.5); hold on;
plot(enei, ext_rm(:,2), 'r--','LineWidth',1.5);
xlabel('Wavelength (nm)'); ylabel('Extinction (nm^2)');
title(sprintf('MATLAB ret mirror sphere (t=%.3fs)', t_rm));
legend('x','y','Location','best'); grid on;
saveas(fig, fullfile(fig_dir,'ret_mirror_matlab.png')); close(fig);

fprintf('[info] MATLAB 06_mirror/sphere done.\n');
