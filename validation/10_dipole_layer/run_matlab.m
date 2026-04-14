%  Validation 10: Dipole + Layer Structure
%  20nm Au sphere on glass, dipole at z=25nm, z-oriented
%  Tests: DipoleStatLayer, DipoleRetLayer, decayrate0

clear; close all;
addpath(genpath('../../'));

%% Setup
epstab = { epsconst(1), epstable('gold.dat'), epsconst(2.25) };
ztab = 0;
opt = layerstructure.options;
layer = layerstructure(epstab, [1, 3], ztab, opt);
op_s = bemoptions('sim', 'stat', 'interp', 'curv', 'layer', layer);
op_r = bemoptions('sim', 'ret',  'interp', 'curv', 'layer', layer);

%  Particle: 20nm Au sphere, 1nm above substrate
p = trisphere(144, 20);
p = shift(p, [0, 0, -min(p.pos(:,3)) + 1]);
p = comparticle(epstab, {p}, [2, 1], 1, op_s);

%  Wavelength grid
enei = linspace(500, 700, 21);

%  Dipole at z=25nm, z-oriented
pt = compoint(p, [0, 0, 25], op_s);

%% 1. DipoleStatLayer
fprintf('=== DipoleStatLayer ===\n');
dip_s = dipole(pt, [0, 0, 1], op_s);
bem_s = bemsolver(p, op_s);

tot_s = zeros(length(enei), 1);
rad_s = zeros(length(enei), 1);
tic;
for i = 1:length(enei)
    sig = bem_s \ dip_s(p, enei(i));
    [t, r] = dip_s.decayrate(sig);
    tot_s(i) = t(3);   % z-dipole is 3rd component
    rad_s(i) = r(3);
    fprintf('  [%d/%d] lambda=%.1f tot=%.6f rad=%.6f\n', i, length(enei), enei(i), tot_s(i), rad_s(i));
end
time_stat = toc;
fprintf('DipoleStatLayer time: %.3f s\n', time_stat);

%% 2. DipoleRetLayer
fprintf('\n=== DipoleRetLayer ===\n');

% Green function table
if ~exist('greentab', 'var') || ~greentab.ismember(layer, enei, {p, pt})
    tab = tabspace(layer, {p, pt}, 'nz', 5);
    greentab = compgreentablayer(layer, tab);
    greentab = set(greentab, enei, op_r, 'waitbar', 0);
end
op_r.greentab = greentab;

dip_r = dipole(pt, [0, 0, 1], op_r);
bem_r = bemsolver(p, op_r);

tot_r = zeros(length(enei), 1);
rad_r = zeros(length(enei), 1);
tic;
for i = 1:length(enei)
    sig = bem_r \ dip_r(p, enei(i));
    [t, r] = dip_r.decayrate(sig);
    tot_r(i) = t(3);
    rad_r(i) = r(3);
    fprintf('  [%d/%d] lambda=%.1f tot=%.6f rad=%.6f\n', i, length(enei), enei(i), tot_r(i), rad_r(i));
end
time_ret = toc;
fprintf('DipoleRetLayer time: %.3f s\n', time_ret);

%% 3. decayrate0 (no particle)
fprintf('\n=== decayrate0 ===\n');

tot0_s = zeros(length(enei), 1);
rad0_s = zeros(length(enei), 1);
tot0_r = zeros(length(enei), 1);
rad0_r = zeros(length(enei), 1);
tic;
for i = 1:length(enei)
    [t0s, r0s] = dip_s.decayrate0(enei(i));
    tot0_s(i) = t0s(3);
    rad0_s(i) = r0s(3);
    [t0r, r0r] = dip_r.decayrate0(enei(i));
    tot0_r(i) = t0r(3);
    rad0_r(i) = r0r(3);
    fprintf('  [%d/%d] lambda=%.1f stat_tot=%.6f ret_tot=%.6f\n', ...
        i, length(enei), enei(i), tot0_s(i), tot0_r(i));
end
time_dr0 = toc;
fprintf('decayrate0 time: %.3f s\n', time_dr0);

%% Save results
data_dir = fullfile(fileparts(mfilename('fullpath')), 'data');
if ~exist(data_dir, 'dir'), mkdir(data_dir); end

% CSV format
fid = fopen(fullfile(data_dir, 'matlab_statlayer.csv'), 'w');
fprintf(fid, 'wavelength_nm,tot,rad\n');
for i = 1:length(enei)
    fprintf(fid, '%.6f,%.10e,%.10e\n', enei(i), tot_s(i), rad_s(i));
end
fclose(fid);

fid = fopen(fullfile(data_dir, 'matlab_retlayer.csv'), 'w');
fprintf(fid, 'wavelength_nm,tot,rad\n');
for i = 1:length(enei)
    fprintf(fid, '%.6f,%.10e,%.10e\n', enei(i), tot_r(i), rad_r(i));
end
fclose(fid);

fid = fopen(fullfile(data_dir, 'matlab_decayrate0.csv'), 'w');
fprintf(fid, 'wavelength_nm,stat_tot,stat_rad,ret_tot,ret_rad\n');
for i = 1:length(enei)
    fprintf(fid, '%.6f,%.10e,%.10e,%.10e,%.10e\n', ...
        enei(i), tot0_s(i), rad0_s(i), tot0_r(i), rad0_r(i));
end
fclose(fid);

fid = fopen(fullfile(data_dir, 'matlab_timing.csv'), 'w');
fprintf(fid, 'test,time_sec\n');
fprintf(fid, 'statlayer,%.6f\n', time_stat);
fprintf(fid, 'retlayer,%.6f\n', time_ret);
fprintf(fid, 'decayrate0,%.6f\n', time_dr0);
fclose(fid);

fprintf('\nResults saved to %s\n', data_dir);
fprintf('Total time: %.1f s\n', time_stat + time_ret + time_dr0);
