%% generate_matlab_data.m
%  EELS validation: generate MATLAB reference data for Python comparison.
%
%  Tests:
%    1. EELSStat: loss probability vs wavelength 450-650nm (21pt), impact=[15,0]
%    2. EELSRet:  same
%    3. Loss map: impact parameter scan at lambda=520nm
%
%  Gold nanosphere, diameter=20nm, trisphere(144,20)

clear; clc;

%% setup
units;
epstab = { epsconst(1), epstable('gold.dat') };
op_s = bemoptions('sim', 'stat', 'interp', 'curv');
op_r = bemoptions('sim', 'ret',  'interp', 'curv');

p_s = comparticle(epstab, { trisphere(144, 20) }, [2, 1], 1, op_s);
p_r = comparticle(epstab, { trisphere(144, 20) }, [2, 1], 1, op_r);

enei = linspace(450, 650, 21);  % wavelength array (nm)
width = 0.5;
vel   = 0.5;  % in units of c

%% ---- Test 1: EELSStat spectrum ----
fprintf('=== Test 1: EELSStat spectrum ===\n');
tic;

bem_s = bemsolver(p_s, op_s);
exc_s = eelsstat(p_s, [15, 0], width, vel, op_s);

psurf_stat = zeros(1, length(enei));
pbulk_stat = zeros(1, length(enei));

for i = 1 : length(enei)
    sig = bem_s \ exc_s(p_s, enei(i));
    [psurf_stat(i), pbulk_stat(i)] = exc_s.loss(sig);
end

t_stat = toc;
fprintf('  Time: %.3f s\n', t_stat);

%% ---- Test 2: EELSRet spectrum ----
fprintf('=== Test 2: EELSRet spectrum ===\n');
tic;

bem_r = bemsolver(p_r, op_r);
exc_r = eelsret(p_r, [15, 0], width, vel, op_r);

psurf_ret = zeros(1, length(enei));
pbulk_ret = zeros(1, length(enei));

for i = 1 : length(enei)
    sig = bem_r \ exc_r(p_r, enei(i));
    [psurf_ret(i), pbulk_ret(i)] = exc_r.loss(sig);
end

t_ret = toc;
fprintf('  Time: %.3f s\n', t_ret);

%% ---- Test 3: Loss map (impact scan at 520nm) ----
fprintf('=== Test 3: Loss map ===\n');
tic;

imp_arr = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25];
lambda_map = 520;  % nm

% Stat map
psurf_map_stat = zeros(length(imp_arr), 1);
pbulk_map_stat = zeros(length(imp_arr), 1);

for j = 1 : length(imp_arr)
    exc_map_s = eelsstat(p_s, [imp_arr(j), 0], width, vel, op_s);
    sig = bem_s \ exc_map_s(p_s, lambda_map);
    [psurf_map_stat(j), pbulk_map_stat(j)] = exc_map_s.loss(sig);
end

% Ret map
psurf_map_ret = zeros(length(imp_arr), 1);
pbulk_map_ret = zeros(length(imp_arr), 1);

for j = 1 : length(imp_arr)
    exc_map_r = eelsret(p_r, [imp_arr(j), 0], width, vel, op_r);
    sig = bem_r \ exc_map_r(p_r, lambda_map);
    [psurf_map_ret(j), pbulk_map_ret(j)] = exc_map_r.loss(sig);
end

t_map = toc;
fprintf('  Time: %.3f s\n', t_map);

%% ---- Save CSVs ----
output_dir = fileparts(mfilename('fullpath'));

% spectrum CSV: wavelength, psurf_stat, pbulk_stat, psurf_ret, pbulk_ret
T1 = table(enei(:), psurf_stat(:), pbulk_stat(:), psurf_ret(:), pbulk_ret(:), ...
    'VariableNames', {'wavelength_nm', 'psurf_stat', 'pbulk_stat', 'psurf_ret', 'pbulk_ret'});
writetable(T1, fullfile(output_dir, 'data', 'matlab_eels_spectrum.csv'));

% map CSV: impact, psurf_stat, pbulk_stat, psurf_ret, pbulk_ret
T2 = table(imp_arr(:), psurf_map_stat(:), pbulk_map_stat(:), psurf_map_ret(:), pbulk_map_ret(:), ...
    'VariableNames', {'impact_nm', 'psurf_stat', 'pbulk_stat', 'psurf_ret', 'pbulk_ret'});
writetable(T2, fullfile(output_dir, 'data', 'matlab_eels_map.csv'));

% timing
T3 = table({'stat'; 'ret'; 'map'}, [t_stat; t_ret; t_map], ...
    'VariableNames', {'test', 'time_s'});
writetable(T3, fullfile(output_dir, 'data', 'matlab_eels_timing.csv'));

fprintf('\nDone. CSVs saved to data/\n');
fprintf('Total time: %.3f s\n', t_stat + t_ret + t_map);
