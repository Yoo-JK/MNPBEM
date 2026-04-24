%% 12_nearfield / sphere — MATLAB
%  BEMStat + BEMRet, trisphere(144, 20) Au, 31x31 grid at y=0, WL=520nm

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

script_dir = fileparts(mfilename('fullpath'));
data_dir   = fullfile(script_dir, 'data');
fig_dir    = fullfile(script_dir, 'figures');
if ~exist(data_dir, 'dir'); mkdir(data_dir); end
if ~exist(fig_dir,  'dir'); mkdir(fig_dir);  end

WL = 520;
GRID_N = 31;
GRID_RANGE = 30;

epstab = {epsconst(1), epstable('gold.dat')};
op_s = bemoptions('sim','stat','interp','curv');
op_r = bemoptions('sim','ret','interp','curv');

[x, z] = meshgrid(linspace(-GRID_RANGE, GRID_RANGE, GRID_N), ...
                  linspace(-GRID_RANGE, GRID_RANGE, GRID_N));
y = zeros(size(x));

%% Stat
fprintf('[info] BEMStat nearfield\n');
p = comparticle(epstab, {trisphere(144, 20)}, [2, 1], 1, op_s);
bem = bemsolver(p, op_s);
exc = planewave([1,0,0], [0,0,1], op_s);
tic;
sig = bem \ exc(p, WL);
mf = meshfield(p, x, y, z, op_s);
[e, ~] = mf(sig);
t_s = toc;
e2_s = sum(abs(e).^2, 3);
fprintf('[info]   stat = %.4f s\n', t_s);

%% Ret
fprintf('[info] BEMRet nearfield\n');
p2 = comparticle(epstab, {trisphere(144, 20)}, [2, 1], 1, op_r);
bem2 = bemsolver(p2, op_r);
exc2 = planewave([1,0,0], [0,0,1], op_r);
tic;
sig2 = bem2 \ exc2(p2, WL);
mf2 = meshfield(p2, x, y, z, op_r);
[e2, ~] = mf2(sig2);
t_r = toc;
e2_r = sum(abs(e2).^2, 3);
fprintf('[info]   ret  = %.4f s\n', t_r);

%% Save (flatten column-major → match Python row-major after reshape)
xr = x(:); zr = z(:); e2sr = e2_s(:); e2rr = e2_r(:);

fid = fopen(fullfile(data_dir,'stat_matlab.csv'),'w');
fprintf(fid,'x_nm,z_nm,e2\n');
for i = 1:length(xr)
    fprintf(fid,'%.10e,%.10e,%.10e\n', xr(i), zr(i), e2sr(i));
end
fclose(fid);

fid = fopen(fullfile(data_dir,'ret_matlab.csv'),'w');
fprintf(fid,'x_nm,z_nm,e2\n');
for i = 1:length(xr)
    fprintf(fid,'%.10e,%.10e,%.10e\n', xr(i), zr(i), e2rr(i));
end
fclose(fid);

fid = fopen(fullfile(data_dir,'matlab_timing.csv'),'w');
fprintf(fid,'case,time_sec\n');
fprintf(fid,'stat,%.6f\n', t_s);
fprintf(fid,'ret,%.6f\n',  t_r);
fclose(fid);

fprintf('[info] MATLAB 12_nearfield/sphere done.\n');
