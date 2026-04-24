%% 05_bemret_layer / sphere — MATLAB
%  20nm Au sphere, 1nm above glass, BEMRetLayer + greentab
%  normal incidence, ext/sca vs 450-750nm (16pt)

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

script_dir = fileparts(mfilename('fullpath'));
data_dir   = fullfile(script_dir, 'data');
fig_dir    = fullfile(script_dir, 'figures');
if ~exist(data_dir, 'dir'); mkdir(data_dir); end
if ~exist(fig_dir,  'dir'); mkdir(fig_dir);  end

epstab = {epsconst(1), epstable('gold.dat'), epsconst(2.25)};
layer = layerstructure(epstab, [1, 3], 0, layerstructure.options);
op = bemoptions('sim','ret','interp','curv','layer',layer);

p = trisphere(144, 20);
p = shift(p, [0, 0, -min(p.pos(:,3)) + 1]);
p = comparticle(epstab, {p}, [2, 1], 1, op);

tab = tabspace(layer, p);
greentab = compgreentablayer(layer, tab);
greentab = set(greentab, linspace(350, 800, 5), op);
op.greentab = greentab;

bem = bemsolver(p, op);
exc = planewave([1,0,0], [0,0,-1], op);

enei = linspace(450, 750, 16);
n = length(enei);
ext = zeros(1,n); sca = zeros(1,n);

fprintf('[info] BEMRetLayer loop\n');
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    sca(i) = exc.sca(sig);
    ext(i) = exc.extinction(sig);
end
t_bem = toc;
absc = ext - sca;
fprintf('[info]   BEM time = %.4f s\n', t_bem);

fid = fopen(fullfile(data_dir,'bemretlayer_matlab.csv'),'w');
fprintf(fid,'wavelength_nm,extinction,scattering,absorption\n');
for i=1:n
    fprintf(fid,'%.10e,%.10e,%.10e,%.10e\n', enei(i), ext(i), sca(i), absc(i));
end
fclose(fid);

fid = fopen(fullfile(data_dir,'matlab_timing.csv'),'w');
fprintf(fid,'case,time_sec\n');
fprintf(fid,'bem,%.6f\n', t_bem);
fclose(fid);

fig = figure('Visible','off','Position',[100 100 800 500]);
plot(enei, ext, 'b-', 'LineWidth',1.5); hold on;
plot(enei, sca, 'r--','LineWidth',1.5);
plot(enei, absc, 'g:','LineWidth',1.5);
xlabel('Wavelength (nm)'); ylabel('Cross section (nm^2)');
title(sprintf('MATLAB BEMRetLayer sphere (t=%.3fs)', t_bem));
legend('ext','sca','abs','Location','best'); grid on;
saveas(fig, fullfile(fig_dir,'bemretlayer_matlab.png')); close(fig);

fprintf('[info] MATLAB 05_bemret_layer/sphere done.\n');
