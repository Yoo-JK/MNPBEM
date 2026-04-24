%% 01_mie / sphere — MATLAB
%  MieStat 20nm Au, MieRet 100nm Au, MieGans [20,10,10] ellipsoid
%  enei = linspace(400, 800, 41)

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

script_dir = fileparts(mfilename('fullpath'));
data_dir   = fullfile(script_dir, 'data');
fig_dir    = fullfile(script_dir, 'figures');
if ~exist(data_dir, 'dir'); mkdir(data_dir); end
if ~exist(fig_dir,  'dir'); mkdir(fig_dir);  end

enei = linspace(400, 800, 41);
timings = containers.Map('KeyType','char','ValueType','double');

%% MieStat 20nm
fprintf('[info] MieStat 20nm Au sphere\n');
tic;
epsin  = epstable('gold.dat');
epsout = epsconst(1);
m = miestat(epsin, epsout, 20);
ext_s = zeros(size(enei)); sca_s = zeros(size(enei)); abs_s = zeros(size(enei));
for i = 1:length(enei)
    ext_s(i) = extinction(m, enei(i));
    sca_s(i) = scattering(m, enei(i));
    abs_s(i) = absorption(m, enei(i));
end
t_s = toc;
timings('miestat') = t_s;
fprintf('  t = %.4f s\n', t_s);

fid = fopen(fullfile(data_dir,'miestat_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,extinction,scattering,absorption\n');
for i=1:length(enei)
    fprintf(fid,'%.10e,%.10e,%.10e,%.10e\n', enei(i), ext_s(i), sca_s(i), abs_s(i));
end
fclose(fid);

fig = figure('Visible','off','Position',[100 100 800 500]);
plot(enei, ext_s, 'b-', 'LineWidth',1.5); hold on;
plot(enei, sca_s, 'r--','LineWidth',1.5);
plot(enei, abs_s, 'g:', 'LineWidth',1.5);
xlabel('Wavelength (nm)'); ylabel('Cross section (nm^2)');
title(sprintf('MATLAB MieStat 20nm Au sphere (t=%.4fs)', t_s));
legend('ext','sca','abs','Location','best'); grid on;
saveas(fig, fullfile(fig_dir,'miestat_matlab.png')); close(fig);

%% MieRet 100nm
fprintf('[info] MieRet 100nm Au sphere\n');
tic;
op_ret = bemoptions('sim','ret');
m = mieret(epsin, epsout, 100, op_ret);
ext_r = zeros(size(enei)); sca_r = zeros(size(enei)); abs_r = zeros(size(enei));
for i = 1:length(enei)
    ext_r(i) = extinction(m, enei(i));
    sca_r(i) = scattering(m, enei(i));
    abs_r(i) = absorption(m, enei(i));
end
t_r = toc;
timings('mieret') = t_r;
fprintf('  t = %.4f s\n', t_r);

fid = fopen(fullfile(data_dir,'mieret_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,extinction,scattering,absorption\n');
for i=1:length(enei)
    fprintf(fid,'%.10e,%.10e,%.10e,%.10e\n', enei(i), ext_r(i), sca_r(i), abs_r(i));
end
fclose(fid);

fig = figure('Visible','off','Position',[100 100 800 500]);
plot(enei, ext_r, 'b-', 'LineWidth',1.5); hold on;
plot(enei, sca_r, 'r--','LineWidth',1.5);
plot(enei, abs_r, 'g:', 'LineWidth',1.5);
xlabel('Wavelength (nm)'); ylabel('Cross section (nm^2)');
title(sprintf('MATLAB MieRet 100nm Au sphere (t=%.4fs)', t_r));
legend('ext','sca','abs','Location','best'); grid on;
saveas(fig, fullfile(fig_dir,'mieret_matlab.png')); close(fig);

%% MieGans [20,10,10]
fprintf('[info] MieGans [20,10,10]nm ellipsoid\n');
tic;
m = miegans(epsin, epsout, [20, 10, 10]);
ext_gx = zeros(size(enei)); ext_gz = zeros(size(enei));
for i = 1:length(enei)
    ext_gx(i) = extinction(m, enei(i), [1,0,0]);
    ext_gz(i) = extinction(m, enei(i), [0,0,1]);
end
t_g = toc;
timings('miegans') = t_g;
fprintf('  t = %.4f s\n', t_g);

fid = fopen(fullfile(data_dir,'miegans_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,extinction_xpol,extinction_zpol\n');
for i=1:length(enei)
    fprintf(fid,'%.10e,%.10e,%.10e\n', enei(i), ext_gx(i), ext_gz(i));
end
fclose(fid);

fig = figure('Visible','off','Position',[100 100 800 500]);
plot(enei, ext_gx, 'b-', 'LineWidth',1.5); hold on;
plot(enei, ext_gz, 'r--','LineWidth',1.5);
xlabel('Wavelength (nm)'); ylabel('Cross section (nm^2)');
title(sprintf('MATLAB MieGans [20,10,10] ellipsoid (t=%.4fs)', t_g));
legend('x-pol','z-pol','Location','best'); grid on;
saveas(fig, fullfile(fig_dir,'miegans_matlab.png')); close(fig);

%% Timing CSV
fid = fopen(fullfile(data_dir,'matlab_timing.csv'),'w');
fprintf(fid,'case,time_sec\n');
fprintf(fid,'miestat,%.6f\n', t_s);
fprintf(fid,'mieret,%.6f\n',  t_r);
fprintf(fid,'miegans,%.6f\n', t_g);
fclose(fid);

fprintf('[info] total matlab = %.4f s\n', t_s + t_r + t_g);
fprintf('[info] MATLAB 01_mie/sphere done.\n');
