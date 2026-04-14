%% Mie Theory Validation Tests
%  Compute extinction, scattering, absorption for 3 sub-tests:
%    1. MieStat: 20nm Au sphere (quasistatic)
%    2. MieRet: 100nm Au sphere (retarded)
%    3. MieGans: [20,10,10]nm ellipsoid, x-pol and z-pol
%
%  Saves results to data/ directory as CSV files.

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

%% Common parameters
enei = linspace(400, 800, 41);
data_dir = fullfile(fileparts(mfilename('fullpath')), 'data');
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

timing = {};

%% 1. MieStat: 20nm Au sphere
fprintf('=== MieStat: 20nm Au sphere ===\n');
tic;
epsin  = epstable('gold.dat');
epsout = epsconst(1);
mie_s  = miestat(epsin, epsout, 20);

ext_s = zeros(size(enei));
sca_s = zeros(size(enei));
abs_s = zeros(size(enei));
for i = 1:length(enei)
    ext_s(i) = extinction(mie_s, enei(i));
    sca_s(i) = scattering(mie_s, enei(i));
    abs_s(i) = absorption(mie_s, enei(i));
end
t_miestat = toc;
fprintf('  Time: %.4f s\n', t_miestat);
timing{end+1, 1} = 'miestat';
timing{end, 2} = t_miestat;

% Save CSV
fid = fopen(fullfile(data_dir, 'miestat_matlab.csv'), 'w');
fprintf(fid, 'wavelength_nm,extinction,scattering,absorption\n');
for i = 1:length(enei)
    fprintf(fid, '%.6f,%.15e,%.15e,%.15e\n', enei(i), ext_s(i), sca_s(i), abs_s(i));
end
fclose(fid);
fprintf('  Saved: miestat_matlab.csv\n');

%% 2. MieRet: 100nm Au sphere
fprintf('=== MieRet: 100nm Au sphere ===\n');
tic;
epsin  = epstable('gold.dat');
epsout = epsconst(1);
op_ret = bemoptions('sim', 'ret');
mie_r  = mieret(epsin, epsout, 100, op_ret);

ext_r = zeros(size(enei));
sca_r = zeros(size(enei));
abs_r = zeros(size(enei));
for i = 1:length(enei)
    ext_r(i) = extinction(mie_r, enei(i));
    sca_r(i) = scattering(mie_r, enei(i));
    abs_r(i) = absorption(mie_r, enei(i));
end
t_mieret = toc;
fprintf('  Time: %.4f s\n', t_mieret);
timing{end+1, 1} = 'mieret';
timing{end, 2} = t_mieret;

% Save CSV
fid = fopen(fullfile(data_dir, 'mieret_matlab.csv'), 'w');
fprintf(fid, 'wavelength_nm,extinction,scattering,absorption\n');
for i = 1:length(enei)
    fprintf(fid, '%.6f,%.15e,%.15e,%.15e\n', enei(i), ext_r(i), sca_r(i), abs_r(i));
end
fclose(fid);
fprintf('  Saved: mieret_matlab.csv\n');

%% 3. MieGans: [20,10,10]nm ellipsoid
fprintf('=== MieGans: [20,10,10]nm ellipsoid ===\n');
tic;
epsin  = epstable('gold.dat');
epsout = epsconst(1);
mie_g  = miegans(epsin, epsout, [20, 10, 10]);

% x-polarization [1,0,0]
ext_gx = zeros(size(enei));
for i = 1:length(enei)
    ext_gx(i) = extinction(mie_g, enei(i), [1, 0, 0]);
end

% z-polarization [0,0,1]
ext_gz = zeros(size(enei));
for i = 1:length(enei)
    ext_gz(i) = extinction(mie_g, enei(i), [0, 0, 1]);
end
t_miegans = toc;
fprintf('  Time: %.4f s\n', t_miegans);
timing{end+1, 1} = 'miegans';
timing{end, 2} = t_miegans;

% Save CSV (extinction only, two polarizations)
fid = fopen(fullfile(data_dir, 'miegans_matlab.csv'), 'w');
fprintf(fid, 'wavelength_nm,extinction_xpol,extinction_zpol\n');
for i = 1:length(enei)
    fprintf(fid, '%.6f,%.15e,%.15e\n', enei(i), ext_gx(i), ext_gz(i));
end
fclose(fid);
fprintf('  Saved: miegans_matlab.csv\n');

%% Save timing
fid = fopen(fullfile(data_dir, 'matlab_timing.csv'), 'w');
fprintf(fid, 'test,time_seconds\n');
for i = 1:size(timing, 1)
    fprintf(fid, '%s,%.6f\n', timing{i, 1}, timing{i, 2});
end
fclose(fid);
fprintf('\n=== Timing saved to matlab_timing.csv ===\n');
fprintf('  miestat: %.4f s\n', t_miestat);
fprintf('  mieret:  %.4f s\n', t_mieret);
fprintf('  miegans: %.4f s\n', t_miegans);
fprintf('  total:   %.4f s\n', t_miestat + t_mieret + t_miegans);
fprintf('\nDone.\n');
