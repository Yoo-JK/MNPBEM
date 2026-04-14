%% NearField (MeshField) validation — MATLAB reference
%  20nm Au sphere, trisphere(144,20), PlaneWave([1,0,0],[0,0,1])
%  xz-plane meshfield -30~30nm, 31x31 grid, lambda=520nm
%  stat + ret modes

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

data_dir = '/home/yoojk20/workspace/MNPBEM/validation/12_nearfield/data';
fig_dir  = '/home/yoojk20/workspace/MNPBEM/validation/12_nearfield/figures';

%% Grid setup
[x, z] = meshgrid(linspace(-30, 30, 31));
y = 0 * x;

%% ============================
%%  1. Quasistatic (stat)
%% ============================
epstab = {epsconst(1), epstable('gold.dat')};
op_s = bemoptions('sim', 'stat', 'interp', 'curv');
p_s = comparticle(epstab, {trisphere(144, 20)}, [2, 1], 1, op_s);
bem_s = bemsolver(p_s, op_s);
exc_s = planewave([1, 0, 0], [0, 0, 1], op_s);

tic;
sig_s = bem_s \ exc_s(p_s, 520);
mf_s = meshfield(p_s, x, y, z, op_s);
e_s = mf_s.field(sig_s);
t_stat = toc;

% |E|^2
enorm_s = sqrt(sum(abs(e_s).^2, 3));
e2_s = enorm_s.^2;

% Save CSV — flatten to (31*31, 4) table: x, z, enorm, e2
T_s = table(x(:), z(:), enorm_s(:), e2_s(:), ...
    'VariableNames', {'x_nm', 'z_nm', 'enorm', 'e2'});
writetable(T_s, fullfile(data_dir, 'matlab_stat.csv'));

% x=0 linecut (column 16, the middle of 31-point grid)
mid = 16;
z_cut = z(:, mid);
enorm_cut_s = enorm_s(:, mid);
T_lc_s = table(z_cut, enorm_cut_s, ...
    'VariableNames', {'z_nm', 'enorm'});
writetable(T_lc_s, fullfile(data_dir, 'matlab_stat_linecut.csv'));

fprintf('[info] MATLAB stat meshfield time: %.4f sec\n', t_stat);

% Plot: stat |E|^2 colormap
figure('Visible', 'off', 'Position', [100, 100, 700, 600]);
imagesc(linspace(-30, 30, 31), linspace(-30, 30, 31), log10(e2_s));
set(gca, 'YDir', 'normal');
colormap('hot');
colorbar;
xlabel('x (nm)');
ylabel('z (nm)');
title(sprintf('MATLAB BEMStat — |E|^2 (log10), \\lambda=520nm, t=%.3fs', t_stat));
saveas(gcf, fullfile(fig_dir, 'stat_matlab.png'));
close;

%% ============================
%%  2. Retarded (ret)
%% ============================
op_r = bemoptions('sim', 'ret', 'interp', 'curv');
p_r = comparticle(epstab, {trisphere(144, 20)}, [2, 1], 1, op_r);
bem_r = bemsolver(p_r, op_r);
exc_r = planewave([1, 0, 0], [0, 0, 1], op_r);

tic;
sig_r = bem_r \ exc_r(p_r, 520);
mf_r = meshfield(p_r, x, y, z, op_r);
e_r = mf_r.field(sig_r);
t_ret = toc;

% |E|^2
enorm_r = sqrt(sum(abs(e_r).^2, 3));
e2_r = enorm_r.^2;

% Save CSV
T_r = table(x(:), z(:), enorm_r(:), e2_r(:), ...
    'VariableNames', {'x_nm', 'z_nm', 'enorm', 'e2'});
writetable(T_r, fullfile(data_dir, 'matlab_ret.csv'));

% x=0 linecut
enorm_cut_r = enorm_r(:, mid);
T_lc_r = table(z_cut, enorm_cut_r, ...
    'VariableNames', {'z_nm', 'enorm'});
writetable(T_lc_r, fullfile(data_dir, 'matlab_ret_linecut.csv'));

fprintf('[info] MATLAB ret meshfield time: %.4f sec\n', t_ret);

% Plot: ret |E|^2 colormap
figure('Visible', 'off', 'Position', [100, 100, 700, 600]);
imagesc(linspace(-30, 30, 31), linspace(-30, 30, 31), log10(e2_r));
set(gca, 'YDir', 'normal');
colormap('hot');
colorbar;
xlabel('x (nm)');
ylabel('z (nm)');
title(sprintf('MATLAB BEMRet — |E|^2 (log10), \\lambda=520nm, t=%.3fs', t_ret));
saveas(gcf, fullfile(fig_dir, 'ret_matlab.png'));
close;

%% Timing CSV
T_time = table({'stat'; 'ret'}, [t_stat; t_ret], ...
    'VariableNames', {'solver', 'time_sec'});
writetable(T_time, fullfile(data_dir, 'matlab_timing.csv'));

fprintf('[info] MATLAB nearfield validation complete.\n');
exit;
