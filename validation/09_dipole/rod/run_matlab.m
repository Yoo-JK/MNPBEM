%% 09_dipole / rod — MATLAB
%  DipoleStat / DipoleRet, trirod(10,40,[15,15,15]) Au, z-dipole at [0,0,25]

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

script_dir = fileparts(mfilename('fullpath'));
data_dir   = fullfile(script_dir, 'data');
fig_dir    = fullfile(script_dir, 'figures');
if ~exist(data_dir, 'dir'); mkdir(data_dir); end
if ~exist(fig_dir,  'dir'); mkdir(fig_dir);  end

epstab = {epsconst(1), epstable('gold.dat')};
op_s = bemoptions('sim','stat','interp','curv');
op_r = bemoptions('sim','ret','interp','curv');

enei = linspace(500, 700, 21);
n = length(enei);
pos = [0, 0, 25];
dipmom = [0, 0, 1];

fprintf('[info] DipoleStat\n');
p_s = comparticle(epstab, {trirod(10, 40, [15,15,15])}, [2, 1], 1, op_s);
pt_s = compoint(p_s, pos, op_s);
dip_s = dipole(pt_s, dipmom, op_s);
bem_s = bemsolver(p_s, op_s);

tot_s = zeros(1,n); rad_s = zeros(1,n);
tic;
for i = 1:n
    sig = bem_s \ dip_s(p_s, enei(i));
    [t, r] = dip_s.decayrate(sig);
    tot_s(i) = t;
    rad_s(i) = r;
end
t_s = toc;
fprintf('[info]   stat = %.4f s\n', t_s);

fprintf('[info] DipoleRet\n');
p_r = comparticle(epstab, {trirod(10, 40, [15,15,15])}, [2, 1], 1, op_r);
pt_r = compoint(p_r, pos, op_r);
dip_r = dipole(pt_r, dipmom, op_r);
bem_r = bemsolver(p_r, op_r);

tot_r = zeros(1,n); rad_r = zeros(1,n);
tic;
for i = 1:n
    sig = bem_r \ dip_r(p_r, enei(i));
    [t, r] = dip_r.decayrate(sig);
    tot_r(i) = t;
    rad_r(i) = r;
end
t_r = toc;
fprintf('[info]   ret  = %.4f s\n', t_r);

fid = fopen(fullfile(data_dir,'stat_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,tot,rad\n');
for i=1:n; fprintf(fid,'%.10e,%.10e,%.10e\n', enei(i), tot_s(i), rad_s(i)); end
fclose(fid);

fid = fopen(fullfile(data_dir,'ret_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,tot,rad\n');
for i=1:n; fprintf(fid,'%.10e,%.10e,%.10e\n', enei(i), tot_r(i), rad_r(i)); end
fclose(fid);

fid = fopen(fullfile(data_dir,'matlab_timing.csv'),'w');
fprintf(fid,'case,time_sec\n');
fprintf(fid,'stat,%.6f\n', t_s);
fprintf(fid,'ret,%.6f\n',  t_r);
fclose(fid);

fprintf('[info] MATLAB 09_dipole/rod done.\n');
