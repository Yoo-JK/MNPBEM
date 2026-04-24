%% 02_bemstat / rod — MATLAB reference
%  BEMStat + PlaneWave([1,0,0;0,0,1]), trirod(10, 40, [15,15,15])
%  Au rod, x-pol + z-pol, ext/sca vs 400-800nm (41pt)

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

script_dir = fileparts(mfilename('fullpath'));
data_dir   = fullfile(script_dir, 'data');
fig_dir    = fullfile(script_dir, 'figures');
if ~exist(data_dir, 'dir'); mkdir(data_dir); end
if ~exist(fig_dir,  'dir'); mkdir(fig_dir);  end

enei = linspace(400, 800, 41);
n = length(enei);

%% Setup
epstab = {epsconst(1), epstable('gold.dat')};
op = bemoptions('sim','stat','interp','curv');
p  = comparticle(epstab, {trirod(10, 40, [15,15,15])}, [2, 1], 1, op);
bem = bemsolver(p, op);
exc = planewave([1,0,0;0,0,1], [0,0,1;1,0,0], op);

ext_x = zeros(1,n); sca_x = zeros(1,n);
ext_z = zeros(1,n); sca_z = zeros(1,n);

%% BEM loop
fprintf('[info] BEMStat rod loop ...\n');
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    e = exc.extinction(sig);
    s = exc.scattering(sig);
    ext_x(i) = e(1); ext_z(i) = e(2);
    sca_x(i) = s(1); sca_z(i) = s(2);
end
t_bem = toc;
fprintf('[info]   BEM time = %.4f s\n', t_bem);

%% Save
fid = fopen(fullfile(data_dir,'bemstat_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,extinction_xpol,scattering_xpol,extinction_zpol,scattering_zpol\n');
for i=1:n
    fprintf(fid,'%.10e,%.10e,%.10e,%.10e,%.10e\n', enei(i), ext_x(i), sca_x(i), ext_z(i), sca_z(i));
end
fclose(fid);

fid = fopen(fullfile(data_dir,'matlab_timing.csv'),'w');
fprintf(fid,'case,time_sec\n');
fprintf(fid,'bem,%.6f\n', t_bem);
fclose(fid);

%% Plot
fig = figure('Visible','off','Position',[100 100 800 500]);
plot(enei, ext_x, 'b-', 'LineWidth',1.5); hold on;
plot(enei, ext_z, 'r--','LineWidth',1.5);
plot(enei, sca_x, 'b:', 'LineWidth',1.0);
plot(enei, sca_z, 'r:', 'LineWidth',1.0);
xlabel('Wavelength (nm)'); ylabel('Cross section (nm^2)');
title(sprintf('MATLAB BEMStat rod (t_{BEM}=%.3fs)', t_bem));
legend('ext xpol','ext zpol','sca xpol','sca zpol','Location','best'); grid on;
saveas(fig, fullfile(fig_dir,'bemstat_matlab.png')); close(fig);

fprintf('[info] MATLAB 02_bemstat/rod done.\n');
