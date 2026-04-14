%% Iterative BEM Solver Validation — MATLAB reference
%  Tests: BEMStatIter vs BEMStat, BEMRetIter vs BEMRet,
%         BEMRetLayerIter vs BEMRetLayer
%  Geometry: 20nm Au sphere (stat/ret), 20nm Au sphere on glass (retlayer)
%  Saves extinction spectra + timing to data/ directory.

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

data_dir = fullfile(fileparts(mfilename('fullpath')), 'data');
fig_dir  = fullfile(fileparts(mfilename('fullpath')), 'figures');
if ~exist(data_dir, 'dir'), mkdir(data_dir); end
if ~exist(fig_dir,  'dir'), mkdir(fig_dir);  end

%% ========================================================================
%  1. BEMStat (direct) vs BEMStatIter — 400-800nm, 41pt
%% ========================================================================
fprintf('=== Stat: direct vs iterative ===\n');

epstab = {epsconst(1), epstable('gold.dat')};
op = bemoptions('sim', 'stat', 'interp', 'curv');
p = comparticle(epstab, {trisphere(144, 20)}, [2, 1], 1, op);
exc = planewave([1, 0, 0], [0, 0, 1], op);
enei = linspace(400, 800, 41);
n = length(enei);

% --- direct ---
bem_d = bemsolver(p, op);
ext_d = zeros(1, n);
tic;
for i = 1:n
    sig = bem_d \ exc(p, enei(i));
    ext_d(i) = exc.extinction(sig);
end
t_stat_d = toc;
fprintf('  direct  : %.4f s\n', t_stat_d);

% --- iterative ---
bem_i = bemiter(p, op);
ext_i = zeros(1, n);
tic;
for i = 1:n
    sig = bem_i \ exc(p, enei(i));
    ext_i(i) = exc.extinction(sig);
end
t_stat_i = toc;
fprintf('  iterative: %.4f s\n', t_stat_i);

% Save CSVs
write_csv(fullfile(data_dir, 'matlab_stat_direct.csv'), enei, ext_d);
write_csv(fullfile(data_dir, 'matlab_stat_iter.csv'),   enei, ext_i);

% Plot
figure('Visible', 'off', 'Position', [100 100 800 500]);
plot(enei, ext_d, 'b-', 'LineWidth', 1.5); hold on;
plot(enei, ext_i, 'r--', 'LineWidth', 1.5);
xlabel('Wavelength (nm)'); ylabel('Extinction (nm^2)');
title(sprintf('MATLAB Stat — direct(%.3fs) vs iter(%.3fs)', t_stat_d, t_stat_i));
legend('direct', 'iterative', 'Location', 'best'); grid on;
saveas(gcf, fullfile(fig_dir, 'stat_matlab.png')); close;

%% ========================================================================
%  2. BEMRet (direct) vs BEMRetIter — 400-800nm, 41pt
%% ========================================================================
fprintf('=== Ret: direct vs iterative ===\n');

op_r = bemoptions('sim', 'ret', 'interp', 'curv');
p_r = comparticle(epstab, {trisphere(144, 20)}, [2, 1], 1, op_r);
exc_r = planewave([1, 0, 0], [0, 0, 1], op_r);

% --- direct ---
bem_rd = bemsolver(p_r, op_r);
ext_rd = zeros(1, n);
tic;
for i = 1:n
    sig = bem_rd \ exc_r(p_r, enei(i));
    ext_rd(i) = exc_r.extinction(sig);
end
t_ret_d = toc;
fprintf('  direct  : %.4f s\n', t_ret_d);

% --- iterative ---
bem_ri = bemiter(p_r, op_r);
ext_ri = zeros(1, n);
tic;
for i = 1:n
    sig = bem_ri \ exc_r(p_r, enei(i));
    ext_ri(i) = exc_r.extinction(sig);
end
t_ret_i = toc;
fprintf('  iterative: %.4f s\n', t_ret_i);

% Save CSVs
write_csv(fullfile(data_dir, 'matlab_ret_direct.csv'), enei, ext_rd);
write_csv(fullfile(data_dir, 'matlab_ret_iter.csv'),   enei, ext_ri);

% Plot
figure('Visible', 'off', 'Position', [100 100 800 500]);
plot(enei, ext_rd, 'b-', 'LineWidth', 1.5); hold on;
plot(enei, ext_ri, 'r--', 'LineWidth', 1.5);
xlabel('Wavelength (nm)'); ylabel('Extinction (nm^2)');
title(sprintf('MATLAB Ret — direct(%.3fs) vs iter(%.3fs)', t_ret_d, t_ret_i));
legend('direct', 'iterative', 'Location', 'best'); grid on;
saveas(gcf, fullfile(fig_dir, 'ret_matlab.png')); close;

%% ========================================================================
%  3. BEMRetLayer (direct) vs BEMRetLayerIter — 450-750nm, 11pt
%% ========================================================================
fprintf('=== RetLayer: direct vs iterative ===\n');

epstab_l = {epsconst(1), epstable('gold.dat'), epsconst(2.25)};
op_l = bemoptions('sim', 'ret', 'interp', 'curv', 'layer', ...
    layerstructure(epstab_l, [1, 3], 0));
layer = op_l.layer;

sp = trisphere(64, 20);
zmin_sp = min(sp.verts(:, 3));
sp = shift(sp, [0, 0, -zmin_sp + 1]);
p_l = comparticle(epstab_l, {sp}, [2, 1], 1, op_l);

exc_l = planewave([1, 0, 0], [0, 0, -1], op_l);
enei_l = linspace(450, 750, 11);
nl = length(enei_l);

% --- direct ---
bem_ld = bemsolver(p_l, op_l);
ext_ld = zeros(1, nl);
tic;
for i = 1:nl
    sig = bem_ld \ exc_l(p_l, enei_l(i));
    ext_ld(i) = exc_l.extinction(sig);
end
t_retl_d = toc;
fprintf('  direct  : %.4f s\n', t_retl_d);

% --- iterative ---
bem_li = bemiter(p_l, op_l);
ext_li = zeros(1, nl);
tic;
for i = 1:nl
    sig = bem_li \ exc_l(p_l, enei_l(i));
    ext_li(i) = exc_l.extinction(sig);
end
t_retl_i = toc;
fprintf('  iterative: %.4f s\n', t_retl_i);

% Save CSVs
write_csv(fullfile(data_dir, 'matlab_retlayer_direct.csv'), enei_l, ext_ld);
write_csv(fullfile(data_dir, 'matlab_retlayer_iter.csv'),   enei_l, ext_li);

% Plot
figure('Visible', 'off', 'Position', [100 100 800 500]);
plot(enei_l, ext_ld, 'b-', 'LineWidth', 1.5); hold on;
plot(enei_l, ext_li, 'r--', 'LineWidth', 1.5);
xlabel('Wavelength (nm)'); ylabel('Extinction (nm^2)');
title(sprintf('MATLAB RetLayer — direct(%.3fs) vs iter(%.3fs)', t_retl_d, t_retl_i));
legend('direct', 'iterative', 'Location', 'best'); grid on;
saveas(gcf, fullfile(fig_dir, 'retlayer_matlab.png')); close;

%% ========================================================================
%  Timing summary
%% ========================================================================
fid = fopen(fullfile(data_dir, 'matlab_timing.csv'), 'w');
fprintf(fid, 'solver,time_seconds\n');
fprintf(fid, 'stat_direct,%.6f\n',     t_stat_d);
fprintf(fid, 'stat_iter,%.6f\n',       t_stat_i);
fprintf(fid, 'ret_direct,%.6f\n',      t_ret_d);
fprintf(fid, 'ret_iter,%.6f\n',        t_ret_i);
fprintf(fid, 'retlayer_direct,%.6f\n', t_retl_d);
fprintf(fid, 'retlayer_iter,%.6f\n',   t_retl_i);
fclose(fid);

fprintf('\n=== Timing summary ===\n');
fprintf('  stat    : direct=%.3fs  iter=%.3fs\n', t_stat_d, t_stat_i);
fprintf('  ret     : direct=%.3fs  iter=%.3fs\n', t_ret_d,  t_ret_i);
fprintf('  retlayer: direct=%.3fs  iter=%.3fs\n', t_retl_d, t_retl_i);
fprintf('\nDone.\n');
exit;


%% ========================================================================
%  Helper: write 2-column CSV (wavelength, extinction)
%% ========================================================================
function write_csv(filepath, enei, ext)
    fid = fopen(filepath, 'w');
    fprintf(fid, 'wavelength_nm,extinction\n');
    for i = 1:length(enei)
        fprintf(fid, '%.6f,%.15e\n', enei(i), ext(i));
    end
    fclose(fid);
end
