%% Eigenmode validation -- MATLAB reference
%  20nm Au sphere, trisphere(144,20), nev=20
%  1) BEMStatEig extinction vs BEMStat direct (400-800nm, 41pt)
%  2) PlasmonMode eigenvalues (nev=10)
%  3) Top 3 plasmon mode surface charges

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

data_dir = '/home/yoojk20/workspace/MNPBEM/validation/07_eigenmode/data';
fig_dir  = '/home/yoojk20/workspace/MNPBEM/validation/07_eigenmode/figures';

%% Setup
epstab = {epsconst(1), epstable('gold.dat')};
op = bemoptions('sim', 'stat', 'interp', 'curv');
p = comparticle(epstab, {trisphere(144, 20)}, [2, 1], 1, op);
exc = planewave([1, 0, 0], [0, 0, 1], op);
enei = linspace(400, 800, 41);
n = length(enei);

%% 1. BEMStatEig extinction spectrum
fprintf('=== BEMStatEig spectrum (nev=20) ===\n');
tic;
bem_eig = bemstateig(p, op, 'nev', 20);
ext_eig = zeros(1, n);
for i = 1:n
    sig = bem_eig \ exc(p, enei(i));
    ext_eig(i) = exc.extinction(sig);
end
t_eig = toc;
fprintf('  Time: %.4f s\n', t_eig);

%% 2. BEMStat direct extinction spectrum
fprintf('=== BEMStat direct spectrum ===\n');
tic;
bem_dir = bemsolver(p, op);
ext_dir = zeros(1, n);
for i = 1:n
    sig = bem_dir \ exc(p, enei(i));
    ext_dir(i) = exc.extinction(sig);
end
t_dir = toc;
fprintf('  Time: %.4f s\n', t_dir);

%% Save extinction CSVs
fid = fopen(fullfile(data_dir, 'matlab_eig_spectrum.csv'), 'w');
fprintf(fid, 'wavelength_nm,extinction\n');
for i = 1:n
    fprintf(fid, '%.6f,%.15e\n', enei(i), ext_eig(i));
end
fclose(fid);

fid = fopen(fullfile(data_dir, 'matlab_dir_spectrum.csv'), 'w');
fprintf(fid, 'wavelength_nm,extinction\n');
for i = 1:n
    fprintf(fid, '%.6f,%.15e\n', enei(i), ext_dir(i));
end
fclose(fid);

%% 3. PlasmonMode eigenvalues (nev=10)
fprintf('=== PlasmonMode eigenvalues (nev=10) ===\n');
[ene, ur, ul] = plasmonmode(p, op, 'nev', 10);
nev_actual = length(ene);

fid = fopen(fullfile(data_dir, 'matlab_eigenvalues.csv'), 'w');
fprintf(fid, 'mode_index,eigenvalue\n');
for i = 1:nev_actual
    fprintf(fid, '%d,%.15e\n', i, ene(i));
end
fclose(fid);

%% 4. Top 3 mode surface charges
fprintf('=== Top 3 mode surface charges ===\n');
nfaces = size(ur, 1);
pos = p.pos;

for k = 1:3
    fid = fopen(fullfile(data_dir, sprintf('matlab_mode%d_charge.csv', k)), 'w');
    fprintf(fid, 'x,y,z,charge_real,charge_imag\n');
    for j = 1:nfaces
        fprintf(fid, '%.15e,%.15e,%.15e,%.15e,%.15e\n', ...
            pos(j,1), pos(j,2), pos(j,3), real(ur(j,k)), imag(ur(j,k)));
    end
    fclose(fid);
end

%% Save timing
fid = fopen(fullfile(data_dir, 'matlab_timing.csv'), 'w');
fprintf(fid, 'solver,time_sec\n');
fprintf(fid, 'bemstateig,%.6f\n', t_eig);
fprintf(fid, 'bemstat_direct,%.6f\n', t_dir);
fclose(fid);

%% Plots
% Eigenmode vs Direct spectrum
figure('Visible', 'off', 'Position', [100, 100, 800, 500]);
plot(enei, ext_eig, 'b-', 'LineWidth', 1.5); hold on;
plot(enei, ext_dir, 'r--', 'LineWidth', 1.5);
xlabel('Wavelength (nm)');
ylabel('Extinction (nm^2)');
title(sprintf('MATLAB: BEMStatEig vs BEMStat (t_{eig}=%.3fs, t_{dir}=%.3fs)', t_eig, t_dir));
legend('Eigenmode (nev=20)', 'Direct', 'Location', 'best');
grid on;
saveas(gcf, fullfile(fig_dir, 'eig_spectrum_matlab.png'));
close;

% Eigenvalue bar chart
figure('Visible', 'off', 'Position', [100, 100, 800, 500]);
bar(1:nev_actual, ene);
xlabel('Mode index');
ylabel('Eigenvalue');
title('MATLAB: Plasmon eigenvalues (nev=10)');
grid on;
saveas(gcf, fullfile(fig_dir, 'eigenvalues_matlab.png'));
close;

fprintf('[info] MATLAB eigenmode validation complete.\n');
fprintf('[info] BEMStatEig time: %.4f s\n', t_eig);
fprintf('[info] BEMStat    time: %.4f s\n', t_dir);
exit;
