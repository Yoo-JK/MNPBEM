%% 03_bemret / sphere — MATLAB reference
%  BEMRet + PlaneWave([1,0,0],[0,0,1]), trisphere(144, 20), Au sphere
%  ext/sca/abs vs 400-800nm (41pt) + MieRet reference

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

script_dir = fileparts(mfilename('fullpath'));
data_dir   = fullfile(script_dir, 'data');
fig_dir    = fullfile(script_dir, 'figures');
if ~exist(data_dir, 'dir'); mkdir(data_dir); end
if ~exist(fig_dir,  'dir'); mkdir(fig_dir);  end

enei = linspace(400, 800, 41);
n = length(enei);

%% Setup
op = bemoptions('sim','ret','interp','curv');
epstab = {epsconst(1), epstable('gold.dat')};
p = comparticle(epstab, {trisphere(144, 20)}, [2, 1], 1, op);
bem = bemsolver(p, op);
exc = planewave([1, 0, 0], [0, 0, 1], op);
mie = miesolver(epstab{2}, epstab{1}, 20, op);

ext = zeros(1,n); sca = zeros(1,n);
mie_ext = zeros(1,n); mie_sca = zeros(1,n);

%% BEM loop
fprintf('[info] BEMRet loop ...\n');
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    ext(i) = exc.ext(sig);
    sca(i) = exc.sca(sig);
end
t_bem = toc;
absc = ext - sca;
fprintf('[info]   BEM time = %.4f s\n', t_bem);

%% Mie
tic;
for i = 1:n
    mie_ext(i) = mie.ext(enei(i));
    mie_sca(i) = mie.sca(enei(i));
end
t_mie = toc;
mie_abs = mie_ext - mie_sca;
fprintf('[info]   Mie time = %.4f s\n', t_mie);

%% Save
fid = fopen(fullfile(data_dir,'bemret_matlab.csv'),'w');
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
title(sprintf('MATLAB BEMRet 20nm Au sphere (t_{BEM}=%.3fs)', t_bem));
legend('BEM ext','BEM sca','MieRet ext','Location','best'); grid on;
saveas(fig, fullfile(fig_dir,'bemret_matlab.png')); close(fig);

fprintf('[info] MATLAB 03_bemret/sphere done.\n');
