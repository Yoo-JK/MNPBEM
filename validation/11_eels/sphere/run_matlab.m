%% 11_eels / sphere — MATLAB
%  EELSStat / EELSRet, 20nm Au sphere, impact=[15,0], width=0.5, vel=0.5

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

script_dir = fileparts(mfilename('fullpath'));
data_dir   = fullfile(script_dir, 'data');
fig_dir    = fullfile(script_dir, 'figures');
if ~exist(data_dir, 'dir'); mkdir(data_dir); end
if ~exist(fig_dir,  'dir'); mkdir(fig_dir);  end

epstab = {epsconst(1), epstable('gold.dat')};
op_s = bemoptions('sim','stat','interp','curv');
op_r = bemoptions('sim','ret','interp','curv');

enei = linspace(450, 650, 21);
n = length(enei);
impact = [15, 0];
width = 0.5;
vel = 0.5;

%% EELSStat
fprintf('[info] EELSStat\n');
p_s = comparticle(epstab, {trisphere(144, 20)}, [2, 1], 1, op_s);
exc_s = electronbeam(p_s, impact, width, vel, op_s);
bem_s = bemsolver(p_s, op_s);

psurf_s = zeros(1,n); pbulk_s = zeros(1,n);
tic;
for i = 1:n
    sig = bem_s \ exc_s(enei(i));
    [ps, pb] = exc_s.loss(sig);
    psurf_s(i) = ps;
    pbulk_s(i) = pb;
end
t_s = toc;
fprintf('[info]   stat = %.4f s\n', t_s);

%% EELSRet
fprintf('[info] EELSRet\n');
p_r = comparticle(epstab, {trisphere(144, 20)}, [2, 1], 1, op_r);
exc_r = electronbeam(p_r, impact, width, vel, op_r);
bem_r = bemsolver(p_r, op_r);

psurf_r = zeros(1,n); pbulk_r = zeros(1,n);
tic;
for i = 1:n
    sig = bem_r \ exc_r(enei(i));
    [ps, pb] = exc_r.loss(sig);
    psurf_r(i) = ps;
    pbulk_r(i) = pb;
end
t_r = toc;
fprintf('[info]   ret  = %.4f s\n', t_r);

%% Save
fid = fopen(fullfile(data_dir,'stat_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,psurf,pbulk\n');
for i=1:n; fprintf(fid,'%.10e,%.10e,%.10e\n', enei(i), psurf_s(i), pbulk_s(i)); end
fclose(fid);

fid = fopen(fullfile(data_dir,'ret_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,psurf,pbulk\n');
for i=1:n; fprintf(fid,'%.10e,%.10e,%.10e\n', enei(i), psurf_r(i), pbulk_r(i)); end
fclose(fid);

fid = fopen(fullfile(data_dir,'matlab_timing.csv'),'w');
fprintf(fid,'case,time_sec\n');
fprintf(fid,'stat,%.6f\n', t_s);
fprintf(fid,'ret,%.6f\n',  t_r);
fclose(fid);

fprintf('[info] MATLAB 11_eels/sphere done.\n');
