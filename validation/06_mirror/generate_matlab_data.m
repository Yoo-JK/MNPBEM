%% Mirror symmetry validation: BEMStatMirror / BEMRetMirror vs full sphere
%  Gold nanosphere, diameter 20nm, 400-800nm, 41pt
%  Full sphere: expanded from comparticlemirror with BEMStat / BEMRet
%  Mirror:      trispheresegment 1/4 sphere + comparticlemirror sym='xy'
%
%  Both full and mirror use the SAME expanded mesh for fair comparison.

clear; close all;

%% Common parameters
epstab = {epsconst(1), epstable('gold.dat')};
op = bemoptions('sim', 'stat', 'interp', 'curv');
enei = linspace(400, 800, 41);

%% Build mirror particle (1/4 sphere, sym='xy')
n = 13;
phi = linspace(0, pi/2, n);
theta = linspace(0, pi, 2*n - 1);
p_seg = trispheresegment(phi, theta, 20);

%% ==================== Quasistatic ====================

% --- Mirror ---
p_mir = comparticlemirror(epstab, {p_seg}, [2, 1], 1, op, 'sym', 'xy');
bem_mir = bemstatmirror(p_mir, op);
exc_mir = planewavestatmirror([1, 0, 0; 0, 1, 0; 0, 0, 1]);

ext_mir_stat = zeros(length(enei), 3);
t_mir_stat = tic;
for i = 1:length(enei)
    sig = bem_mir \ exc_mir(p_mir, enei(i));
    ext_mir_stat(i, :) = extinction(exc_mir, sig);
end
t_mir_stat = toc(t_mir_stat);

% --- Full (same expanded mesh) ---
p_full = full(p_mir);
bem_full = bemstat(p_full, op);
exc_full = planewavestat([1, 0, 0; 0, 1, 0; 0, 0, 1]);

ext_full_stat = zeros(length(enei), 3);
t_full_stat = tic;
for i = 1:length(enei)
    sig = bem_full \ exc_full(p_full, enei(i));
    ext_full_stat(i, :) = extinction(exc_full, sig);
end
t_full_stat = toc(t_full_stat);

fprintf('Stat full: %.2f s, mirror: %.2f s, speedup: %.2fx\n', ...
    t_full_stat, t_mir_stat, t_full_stat / t_mir_stat);

%% ==================== Retarded ====================

op_ret = bemoptions('sim', 'ret', 'interp', 'curv');

% --- Mirror ---
p_mir_ret = comparticlemirror(epstab, {p_seg}, [2, 1], 1, op_ret, 'sym', 'xy');
bem_mir_ret = bemretmirror(p_mir_ret, op_ret);
exc_mir_ret = planewaveretmirror([1, 0, 0; 0, 1, 0], [0, 0, 1; 0, 0, 1]);

ext_mir_ret = zeros(length(enei), 2);
t_mir_ret = tic;
for i = 1:length(enei)
    sig = bem_mir_ret \ exc_mir_ret(p_mir_ret, enei(i));
    ext_mir_ret(i, :) = extinction(exc_mir_ret, sig);
end
t_mir_ret = toc(t_mir_ret);

% --- Full (same expanded mesh) ---
p_full_ret = full(p_mir_ret);
bem_full_ret = bemret(p_full_ret, op_ret);
exc_full_ret = planewaveret([1, 0, 0; 0, 1, 0], [0, 0, 1; 0, 0, 1]);

ext_full_ret = zeros(length(enei), 2);
t_full_ret = tic;
for i = 1:length(enei)
    sig = bem_full_ret \ exc_full_ret(p_full_ret, enei(i));
    ext_full_ret(i, :) = extinction(exc_full_ret, sig);
end
t_full_ret = toc(t_full_ret);

fprintf('Ret full: %.2f s, mirror: %.2f s, speedup: %.2fx\n', ...
    t_full_ret, t_mir_ret, t_full_ret / t_mir_ret);

%% ==================== Save CSVs ====================

% Quasistatic
T = table(enei', ext_full_stat(:,1), ext_full_stat(:,2), ext_full_stat(:,3), ...
    'VariableNames', {'wavelength_nm', 'ext_x', 'ext_y', 'ext_z'});
writetable(T, 'data/matlab_full_stat.csv');

T = table(enei', ext_mir_stat(:,1), ext_mir_stat(:,2), ext_mir_stat(:,3), ...
    'VariableNames', {'wavelength_nm', 'ext_x', 'ext_y', 'ext_z'});
writetable(T, 'data/matlab_mirror_stat.csv');

% Retarded
T = table(enei', ext_full_ret(:,1), ext_full_ret(:,2), ...
    'VariableNames', {'wavelength_nm', 'ext_x', 'ext_y'});
writetable(T, 'data/matlab_full_ret.csv');

T = table(enei', ext_mir_ret(:,1), ext_mir_ret(:,2), ...
    'VariableNames', {'wavelength_nm', 'ext_x', 'ext_y'});
writetable(T, 'data/matlab_mirror_ret.csv');

% Timing
T = table({'stat_full'; 'stat_mirror'; 'ret_full'; 'ret_mirror'}, ...
    [t_full_stat; t_mir_stat; t_full_ret; t_mir_ret], ...
    'VariableNames', {'method', 'time_s'});
writetable(T, 'data/matlab_timing.csv');

fprintf('Done. CSVs saved to data/\n');
