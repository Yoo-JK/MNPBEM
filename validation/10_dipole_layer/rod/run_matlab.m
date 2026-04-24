%% 10_dipole_layer / rod — MATLAB
%  DipoleStatLayer / DipoleRetLayer, rod(10,40,[15,15,15]) 1nm above glass
%  z-dipole at [0,0,45]

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

script_dir = fileparts(mfilename('fullpath'));
data_dir   = fullfile(script_dir, 'data');
fig_dir    = fullfile(script_dir, 'figures');
if ~exist(data_dir, 'dir'); mkdir(data_dir); end
if ~exist(fig_dir,  'dir'); mkdir(fig_dir);  end

epstab = {epsconst(1), epstable('gold.dat'), epsconst(2.25)};
layer = layerstructure(epstab, [1, 3], 0, layerstructure.options);
op_s = bemoptions('sim','stat','interp','curv','layer',layer);
op_r = bemoptions('sim','ret','interp','curv','layer',layer);

enei = linspace(500, 700, 21);
n = length(enei);
pos = [0, 0, 45];
dipmom = [0, 0, 1];

fprintf('[info] DipoleStatLayer\n');
p = trirod(10, 40, [15,15,15]);
p = shift(p, [0, 0, -min(p.pos(:,3)) + 1]);
p_s = comparticle(epstab, {p}, [2, 1], 1, op_s);
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

fprintf('[info] DipoleRetLayer (greentab setup)\n');
p_r = comparticle(epstab, {p}, [2, 1], 1, op_r);
pt_r = compoint(p_r, pos, op_r);
tab = tabspace(layer, {p_r, pt_r}, 'nz', 5);
greentab = compgreentablayer(layer, tab);
greentab = set(greentab, enei, op_r, 'waitbar', 0);
op_r.greentab = greentab;

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

fprintf('[info] MATLAB 10_dipole_layer/rod done.\n');
