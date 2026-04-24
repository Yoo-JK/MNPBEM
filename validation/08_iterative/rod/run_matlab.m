%% 08_iterative / rod — MATLAB
%  BEMStatIter vs BEMStat + BEMRetIter vs BEMRet
%  trirod(10, 40, [15,15,15]) Au, 400-800nm (41pt)

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

script_dir = fileparts(mfilename('fullpath'));
data_dir   = fullfile(script_dir, 'data');
fig_dir    = fullfile(script_dir, 'figures');
if ~exist(data_dir, 'dir'); mkdir(data_dir); end
if ~exist(fig_dir,  'dir'); mkdir(fig_dir);  end

epstab = {epsconst(1), epstable('gold.dat')};
op_s  = bemoptions('sim','stat','interp','curv');
op_si = bemoptions('sim','stat','interp','curv');
op_si.iter = bemiter.options;
op_r  = bemoptions('sim','ret','interp','curv');
op_ri = bemoptions('sim','ret','interp','curv');
op_ri.iter = bemiter.options;

mesh = @() trirod(10, 40, [15,15,15]);

enei = linspace(400, 800, 41);
n = length(enei);

fprintf('[info] BEMStat direct\n');
p = comparticle(epstab, {mesh()}, [2, 1], 1, op_s);
bem = bemsolver(p, op_s);
exc = planewave([1,0,0], [0,0,1], op_s);
ext_sd = zeros(1,n);
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    ext_sd(i) = exc.extinction(sig);
end
t_sd = toc;
fprintf('[info]   direct = %.4f s\n', t_sd);

fprintf('[info] BEMStat iterative\n');
p = comparticle(epstab, {mesh()}, [2, 1], 1, op_si);
bem = bemsolver(p, op_si);
exc = planewave([1,0,0], [0,0,1], op_si);
ext_si = zeros(1,n);
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    ext_si(i) = exc.extinction(sig);
end
t_si = toc;
fprintf('[info]   iter   = %.4f s\n', t_si);

fprintf('[info] BEMRet direct\n');
p = comparticle(epstab, {mesh()}, [2, 1], 1, op_r);
bem = bemsolver(p, op_r);
exc = planewave([1,0,0], [0,0,1], op_r);
ext_rd = zeros(1,n);
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    ext_rd(i) = exc.ext(sig);
end
t_rd = toc;
fprintf('[info]   direct = %.4f s\n', t_rd);

fprintf('[info] BEMRet iterative\n');
p = comparticle(epstab, {mesh()}, [2, 1], 1, op_ri);
bem = bemsolver(p, op_ri);
exc = planewave([1,0,0], [0,0,1], op_ri);
ext_ri = zeros(1,n);
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    ext_ri(i) = exc.ext(sig);
end
t_ri = toc;
fprintf('[info]   iter   = %.4f s\n', t_ri);

%% Save
fid = fopen(fullfile(data_dir,'stat_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,ext_direct,ext_iter\n');
for i=1:n; fprintf(fid,'%.10e,%.10e,%.10e\n', enei(i), ext_sd(i), ext_si(i)); end
fclose(fid);

fid = fopen(fullfile(data_dir,'ret_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,ext_direct,ext_iter\n');
for i=1:n; fprintf(fid,'%.10e,%.10e,%.10e\n', enei(i), ext_rd(i), ext_ri(i)); end
fclose(fid);

fid = fopen(fullfile(data_dir,'matlab_timing.csv'),'w');
fprintf(fid,'case,time_sec\n');
fprintf(fid,'stat_direct,%.6f\n', t_sd);
fprintf(fid,'stat_iter,%.6f\n',   t_si);
fprintf(fid,'ret_direct,%.6f\n',  t_rd);
fprintf(fid,'ret_iter,%.6f\n',    t_ri);
fclose(fid);

fprintf('[info] MATLAB 08_iterative/rod done.\n');
