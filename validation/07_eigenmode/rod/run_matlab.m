%% 07_eigenmode / rod — MATLAB
%  BEMStatEig vs BEMStat, trirod(10,40,[15,15,15]) Au, 400-800nm (41pt)

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

script_dir = fileparts(mfilename('fullpath'));
data_dir   = fullfile(script_dir, 'data');
fig_dir    = fullfile(script_dir, 'figures');
if ~exist(data_dir, 'dir'); mkdir(data_dir); end
if ~exist(fig_dir,  'dir'); mkdir(fig_dir);  end

epstab = {epsconst(1), epstable('gold.dat')};
op = bemoptions('sim','stat','interp','curv');
p = comparticle(epstab, {trirod(10, 40, [15,15,15])}, [2, 1], 1, op);

enei = linspace(400, 800, 41);
n = length(enei);

fprintf('[info] BEMStatEig (nev=20)\n');
op_eig = op;
op_eig.nev = 20;
bem_e = bemstateig(p, op_eig);
exc = planewave([1,0,0], [0,0,1], op);

ext_e = zeros(1, n);
tic;
for i = 1:n
    sig = bem_e \ exc(p, enei(i));
    ext_e(i) = exc.extinction(sig);
end
t_e = toc;
fprintf('[info]   eig time = %.4f s\n', t_e);

fprintf('[info] BEMStat direct\n');
bem_d = bemstat(p, op);
ext_d = zeros(1, n);
tic;
for i = 1:n
    sig = bem_d \ exc(p, enei(i));
    ext_d(i) = exc.extinction(sig);
end
t_d = toc;
fprintf('[info]   direct time = %.4f s\n', t_d);

fid = fopen(fullfile(data_dir,'eig_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,extinction\n');
for i=1:n; fprintf(fid,'%.10e,%.10e\n', enei(i), ext_e(i)); end
fclose(fid);

fid = fopen(fullfile(data_dir,'dir_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,extinction\n');
for i=1:n; fprintf(fid,'%.10e,%.10e\n', enei(i), ext_d(i)); end
fclose(fid);

fid = fopen(fullfile(data_dir,'matlab_timing.csv'),'w');
fprintf(fid,'case,time_sec\n');
fprintf(fid,'eig,%.6f\n', t_e);
fprintf(fid,'dir,%.6f\n', t_d);
fclose(fid);

fig = figure('Visible','off','Position',[100 100 800 500]);
plot(enei, ext_d, 'b-', 'LineWidth',1.5); hold on;
plot(enei, ext_e, 'r--','LineWidth',1.5);
xlabel('Wavelength (nm)'); ylabel('Extinction (nm^2)');
title(sprintf('MATLAB eig/direct rod (eig=%.3fs, dir=%.3fs)', t_e, t_d));
legend('direct','eig(nev=20)','Location','best'); grid on;
saveas(fig, fullfile(fig_dir,'eigenmode_matlab.png')); close(fig);

fprintf('[info] MATLAB 07_eigenmode/rod done.\n');
