%% Step 1 Validation: Compare Python vs MATLAB Material Functions
%  This script generates reference data from MATLAB to validate Python implementation

clear all;
close all;

fprintf('======================================================================\n');
fprintf('Step 1: MATLAB Reference Data Generation\n');
fprintf('======================================================================\n');

%% Add MATLAB MNPBEM to path
addpath(genpath('.'));

%% Test 1: EpsConst (Vacuum)
fprintf('\n----------------------------------------------------------------------\n');
fprintf('Test 1: EpsConst (Vacuum)\n');
fprintf('----------------------------------------------------------------------\n');

eps_vacuum = epsconst(1.0);
wavelengths = [400, 500, 600, 700];

fprintf('At λ = [400, 500, 600, 700] nm:\n');
for i = 1:length(wavelengths)
    [eps_val, k_val] = eps_vacuum(wavelengths(i));
    fprintf('  λ = %g nm: ε = %.6f, k = %.8f 1/nm\n', wavelengths(i), eps_val, k_val);
end

%% Test 2: EpsConst (Water)
fprintf('\n----------------------------------------------------------------------\n');
fprintf('Test 2: EpsConst (Water, n=1.33)\n');
fprintf('----------------------------------------------------------------------\n');

eps_water = epsconst(1.33^2);
[eps_val, k_val] = eps_water(500);
fprintf('At λ = 500 nm:\n');
fprintf('  ε = %.6f, k = %.8f 1/nm\n', eps_val, k_val);

%% Test 3: EpsTable (Gold)
fprintf('\n----------------------------------------------------------------------\n');
fprintf('Test 3: EpsTable (Gold from Johnson & Christy)\n');
fprintf('----------------------------------------------------------------------\n');

eps_gold = epstable('Material/@epstable/gold.dat');
test_wl = [400, 500, 600, 700];

fprintf('Dielectric function at visible wavelengths:\n');
fprintf('%-10s %-15s %-15s %-15s\n', 'λ (nm)', 'ε (real)', 'ε (imag)', 'k (1/nm)');
fprintf('%s\n', repmat('-', 1, 60));

for i = 1:length(test_wl)
    [eps_val, k_val] = eps_gold(test_wl(i));
    fprintf('%-10.1f %-15.6f %-15.6f %-15.8f\n', ...
        test_wl(i), real(eps_val), imag(eps_val), real(k_val));
end

% Save data to file for Python comparison
data_struct.vacuum_eps = zeros(length(wavelengths), 1);
data_struct.vacuum_k = zeros(length(wavelengths), 1);
data_struct.water_eps = 0;
data_struct.water_k = 0;
data_struct.gold_eps = zeros(length(test_wl), 1);
data_struct.gold_k = zeros(length(test_wl), 1);
data_struct.wavelengths = wavelengths';
data_struct.test_wl = test_wl';

for i = 1:length(wavelengths)
    [data_struct.vacuum_eps(i), data_struct.vacuum_k(i)] = eps_vacuum(wavelengths(i));
end

[data_struct.water_eps, data_struct.water_k] = eps_water(500);

for i = 1:length(test_wl)
    [data_struct.gold_eps(i), data_struct.gold_k(i)] = eps_gold(test_wl(i));
end

% Save to MAT file
save('step1_matlab_reference.mat', '-struct', 'data_struct');
fprintf('\nReference data saved to: step1_matlab_reference.mat\n');

%% Test 4: Units
fprintf('\n----------------------------------------------------------------------\n');
fprintf('Test 4: Unit Conversion\n');
fprintf('----------------------------------------------------------------------\n');

units;
fprintf('Conversion factor: eV2nm = %.6f nm·eV\n', eV2nm);

fprintf('\n======================================================================\n');
fprintf('MATLAB Reference Data Generation Complete!\n');
fprintf('======================================================================\n');
