%% BEMStat sphere validation — MATLAB reference
%  20nm Au sphere, trisphere(144,20), PlaneWaveStat([1,0,0])
%  ext + sca + abs vs 400-800nm (41pt)
%  MieStat overlay (analytical reference)

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

%% Setup
epstab = {epsconst(1), epstable('gold.dat')};
op = bemoptions('sim', 'stat', 'interp', 'curv');
p = comparticle(epstab, {trisphere(144, 20)}, [2, 1], 1, op);
bem = bemsolver(p, op);
exc = planewave([1, 0, 0], [0, 0, 1], op);
mie = miestat(epstab{2}, epstab{1}, 20, op);

enei = linspace(400, 800, 41);
n = length(enei);

ext = zeros(1, n);
sca = zeros(1, n);
absc = zeros(1, n);
mie_ext = zeros(1, n);
mie_sca = zeros(1, n);
mie_abs = zeros(1, n);

%% BEM solve loop with timing
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    ext(i) = exc.extinction(sig);
    sca(i) = exc.scattering(sig);
    absc(i) = ext(i) - sca(i);
end
t_bem = toc;

%% Mie solve loop with timing
tic;
for i = 1:n
    mie_ext(i) = mie.extinction(enei(i));
    mie_sca(i) = mie.scattering(enei(i));
    mie_abs(i) = mie_ext(i) - mie_sca(i);
end
t_mie = toc;

%% Save CSVs
data_dir = '/home/yoojk20/workspace/MNPBEM/validation/02_bemstat_sphere/data';

% BEM results
T = table(enei', ext', sca', absc', ...
    'VariableNames', {'wavelength_nm', 'extinction', 'scattering', 'absorption'});
writetable(T, fullfile(data_dir, 'matlab_bemstat.csv'));

% Mie results
T2 = table(enei', mie_ext', mie_sca', mie_abs', ...
    'VariableNames', {'wavelength_nm', 'extinction', 'scattering', 'absorption'});
writetable(T2, fullfile(data_dir, 'matlab_mie.csv'));

% Timing
T3 = table({'BEM'; 'Mie'}, [t_bem; t_mie], ...
    'VariableNames', {'solver', 'time_sec'});
writetable(T3, fullfile(data_dir, 'matlab_timing.csv'));

fprintf('[info] MATLAB BEM solve time: %.4f sec\n', t_bem);
fprintf('[info] MATLAB Mie solve time: %.4f sec\n', t_mie);

%% Plot: BEM ext+sca + Mie ext overlay
fig_dir = '/home/yoojk20/workspace/MNPBEM/validation/02_bemstat_sphere/figures';

figure('Visible', 'off', 'Position', [100, 100, 800, 500]);
plot(enei, ext, 'b-', 'LineWidth', 1.5); hold on;
plot(enei, sca, 'r--', 'LineWidth', 1.5);
plot(enei, mie_ext, 'ko', 'MarkerSize', 4);
xlabel('Wavelength (nm)');
ylabel('Cross section (nm^2)');
title(sprintf('MATLAB BEMStat — 20nm Au sphere (t_{BEM}=%.3fs)', t_bem));
legend('BEM ext', 'BEM sca', 'Mie ext', 'Location', 'best');
grid on;
saveas(gcf, fullfile(fig_dir, 'bemstat_matlab.png'));
close;

fprintf('[info] MATLAB validation complete.\n');
exit;
