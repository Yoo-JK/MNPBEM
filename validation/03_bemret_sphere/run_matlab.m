%% BEMRet Sphere Validation - MATLAB
%  BEMRet + PlaneWaveRet([1,0,0],[0,0,1]), 20nm Au sphere, trisphere(144,20)
%  Computes extinction, scattering, absorption cross sections (400-800nm, 41pt)
%  Also computes MieRet analytical reference and timing.

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

output_dir = '/home/yoojk20/workspace/MNPBEM/validation/03_bemret_sphere/data';

%% Parameters
diameter = 20;
nface = 144;
enei = linspace(400, 800, 41);

%% Setup
op = bemoptions('sim', 'ret', 'interp', 'curv');
epstab = {epsconst(1), epstable('gold.dat')};
sp = trisphere(nface, diameter);
p = comparticle(epstab, {sp}, [2, 1], 1, op);

%% BEMRet simulation
fprintf('Running BEMRet simulation...\n');
tic;

bem = bemsolver(p, op);
exc = planewave([1, 0, 0], [0, 0, 1], op);

ext_bem = zeros(size(enei));
sca_bem = zeros(size(enei));

for ien = 1:length(enei)
    sig = bem \ exc(p, enei(ien));
    ext_bem(ien) = exc.ext(sig);
    sca_bem(ien) = exc.sca(sig);
end

abs_bem = ext_bem - sca_bem;
time_bem = toc;
fprintf('  BEMRet done in %.3f s\n', time_bem);

%% MieRet simulation (analytical reference)
fprintf('Running MieRet simulation...\n');
tic;

mie = miesolver(epstab{2}, epstab{1}, diameter, op);

ext_mie = zeros(size(enei));
sca_mie = zeros(size(enei));

for ien = 1:length(enei)
    ext_mie(ien) = mie.ext(enei(ien));
    sca_mie(ien) = mie.sca(enei(ien));
end

abs_mie = ext_mie - sca_mie;
time_mie = toc;
fprintf('  MieRet done in %.3f s\n', time_mie);

%% Save BEMRet CSV
fid = fopen(fullfile(output_dir, 'matlab_bemret.csv'), 'w');
fprintf(fid, 'wavelength,extinction,scattering,absorption\n');
for i = 1:length(enei)
    fprintf(fid, '%.6f,%.10e,%.10e,%.10e\n', enei(i), ext_bem(i), sca_bem(i), abs_bem(i));
end
fclose(fid);

%% Save MieRet CSV
fid = fopen(fullfile(output_dir, 'matlab_mie.csv'), 'w');
fprintf(fid, 'wavelength,extinction,scattering,absorption\n');
for i = 1:length(enei)
    fprintf(fid, '%.6f,%.10e,%.10e,%.10e\n', enei(i), ext_mie(i), sca_mie(i), abs_mie(i));
end
fclose(fid);

%% Save timing CSV
fid = fopen(fullfile(output_dir, 'matlab_timing.csv'), 'w');
fprintf(fid, 'method,time_seconds\n');
fprintf(fid, 'bemret,%.6f\n', time_bem);
fprintf(fid, 'mieret,%.6f\n', time_mie);
fclose(fid);

fprintf('\nMATLAB results saved to %s\n', output_dir);
fprintf('  matlab_bemret.csv\n');
fprintf('  matlab_mie.csv\n');
fprintf('  matlab_timing.csv\n');
