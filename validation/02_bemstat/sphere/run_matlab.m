%% 02_bemstat / sphere — MATLAB reference
%  BEMStat + PlaneWave([1,0,0]), trisphere(144, 20), Au sphere
%  ext/sca/abs vs enei = linspace(400,800,41)
%  Mie analytical reference (same diameter)

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
p = comparticle(epstab, {trisphere(144, 20)}, [2, 1], 1, op);
bem = bemsolver(p, op);
exc = planewave([1, 0, 0], [0, 0, 1], op);
mie = miestat(epstab{2}, epstab{1}, 20, op);

ext = zeros(1, n); sca = zeros(1, n); absc = zeros(1, n);
mie_ext = zeros(1, n); mie_sca = zeros(1, n); mie_abs = zeros(1, n);

%% BEM
fprintf('[info] BEMStat loop ...\n');
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    ext(i)  = exc.extinction(sig);
    sca(i)  = exc.scattering(sig);
    absc(i) = ext(i) - sca(i);
end
t_bem = toc;
fprintf('[info]   BEM time = %.4f s\n', t_bem);

%% Mie
tic;
for i = 1:n
    mie_ext(i) = mie.extinction(enei(i));
    mie_sca(i) = mie.scattering(enei(i));
    mie_abs(i) = mie_ext(i) - mie_sca(i);
end
t_mie = toc;
fprintf('[info]   Mie time = %.4f s\n', t_mie);

%% Save
fid = fopen(fullfile(data_dir,'bemstat_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,extinction,scattering,absorption\n');
for i=1:n
    fprintf(fid,'%.10e,%.10e,%.10e,%.10e\n', enei(i), ext(i), sca(i), absc(i));
end
fclose(fid);

fid = fopen(fullfile(data_dir,'mie_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,extinction,scattering,absorption\n');
for i=1:n
    fprintf(fid,'%.10e,%.10e,%.10e,%.10e\n', enei(i), mie_ext(i), mie_sca(i), mie_abs(i));
end
fclose(fid);

fid = fopen(fullfile(data_dir,'matlab_timing.csv'),'w');
fprintf(fid,'case,time_sec\n');
fprintf(fid,'bem,%.6f\n', t_bem);
fprintf(fid,'mie,%.6f\n', t_mie);
fclose(fid);

%% Plot
fig = figure('Visible','off','Position',[100 100 800 500]);
plot(enei, ext, 'b-', 'LineWidth',1.5); hold on;
plot(enei, sca, 'r--','LineWidth',1.5);
plot(enei, mie_ext, 'ko','MarkerSize',4);
xlabel('Wavelength (nm)'); ylabel('Cross section (nm^2)');
title(sprintf('MATLAB BEMStat 20nm Au sphere (t_{BEM}=%.3fs)', t_bem));
legend('BEM ext','BEM sca','Mie ext','Location','best'); grid on;
saveas(fig, fullfile(fig_dir,'bemstat_matlab.png')); close(fig);

fprintf('[info] MATLAB 02_bemstat/sphere done.\n');
