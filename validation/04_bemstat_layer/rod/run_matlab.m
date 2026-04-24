%% 04_bemstat_layer / rod — MATLAB
%  trirod(10, 40, [15,15,15]) Au, 1nm above glass substrate (eps=2.25)
%  Normal (theta=0) + Oblique (theta=45, TM)
%  ext/sca vs 400-800nm (41pt)

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

script_dir = fileparts(mfilename('fullpath'));
data_dir   = fullfile(script_dir, 'data');
fig_dir    = fullfile(script_dir, 'figures');
if ~exist(data_dir, 'dir'); mkdir(data_dir); end
if ~exist(fig_dir,  'dir'); mkdir(fig_dir);  end

epstab = {epsconst(1), epstable('gold.dat'), epsconst(2.25)};
layer = layerstructure(epstab, [1, 3], 0, layerstructure.options);
op = bemoptions('sim','stat','interp','curv','layer',layer);

p = trirod(10, 40, [15,15,15]);
p = shift(p, [0, 0, -min(p.pos(:,3)) + 1]);
p = comparticle(epstab, {p}, [2, 1], 1, op);

enei = linspace(400, 800, 41);
n = length(enei);

%% Normal
fprintf('[info] normal (theta=0)\n');
pol_n = [1, 0, 0]; dir_n = [0, 0, -1];
bem = bemsolver(p, op);
exc = planewave(pol_n, dir_n, op);
ext_n = zeros(1,n); sca_n = zeros(1,n);
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    sca_n(i) = exc.sca(sig);
    ext_n(i) = exc.extinction(sig);
end
t_normal = toc;
fprintf('[info]   normal time = %.4f s\n', t_normal);

%% Oblique
fprintf('[info] oblique (theta=45, TM)\n');
theta = pi/4;
pol_o = [cos(theta), 0, sin(theta)];
dir_o = [sin(theta), 0, -cos(theta)];
exc2 = planewave(pol_o, dir_o, op);
ext_o = zeros(1,n); sca_o = zeros(1,n);
tic;
for i = 1:n
    sig = bem \ exc2(p, enei(i));
    sca_o(i) = exc2.sca(sig);
    ext_o(i) = exc2.extinction(sig);
end
t_oblique = toc;
fprintf('[info]   oblique time = %.4f s\n', t_oblique);

%% Save
fid = fopen(fullfile(data_dir,'normal_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,extinction,scattering\n');
for i=1:n
    fprintf(fid,'%.10e,%.10e,%.10e\n', enei(i), ext_n(i), sca_n(i));
end
fclose(fid);

fid = fopen(fullfile(data_dir,'oblique_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,extinction,scattering\n');
for i=1:n
    fprintf(fid,'%.10e,%.10e,%.10e\n', enei(i), ext_o(i), sca_o(i));
end
fclose(fid);

fid = fopen(fullfile(data_dir,'matlab_timing.csv'),'w');
fprintf(fid,'case,time_sec\n');
fprintf(fid,'normal,%.6f\n',  t_normal);
fprintf(fid,'oblique,%.6f\n', t_oblique);
fclose(fid);

%% Plot
fig = figure('Visible','off','Position',[100 100 800 500]);
plot(enei, ext_n, 'b-', 'LineWidth',1.5); hold on;
plot(enei, sca_n, 'r--','LineWidth',1.5);
xlabel('Wavelength (nm)'); ylabel('Cross section (nm^2)');
title(sprintf('MATLAB normal rod (t=%.3fs)', t_normal));
legend('ext','sca','Location','best'); grid on;
saveas(fig, fullfile(fig_dir,'normal_matlab.png')); close(fig);

fig = figure('Visible','off','Position',[100 100 800 500]);
plot(enei, ext_o, 'b-', 'LineWidth',1.5); hold on;
plot(enei, sca_o, 'r--','LineWidth',1.5);
xlabel('Wavelength (nm)'); ylabel('Cross section (nm^2)');
title(sprintf('MATLAB oblique rod (t=%.3fs)', t_oblique));
legend('ext','sca','Location','best'); grid on;
saveas(fig, fullfile(fig_dir,'oblique_matlab.png')); close(fig);

fprintf('[info] MATLAB 04_bemstat_layer/rod done.\n');
