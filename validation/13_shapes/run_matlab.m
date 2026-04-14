%% 13_shapes: BEMStat extinction spectrum for 7 particle shapes
%  All shapes: Au nanoparticle in vacuum, BEMStat + PlaneWaveStat([1,0,0])
%  ext vs 400-800nm (41pt), timing for each shape.
%
%  Shapes:
%    1. trisphere(144, 20)         — sphere
%    2. trirod(10, 40, [15,15,15]) — nanorod (x-pol + z-pol)
%    3. tricube(10, 20)            — nanocube
%    4. tritorus(15, 5, [20,20])   — torus
%    5. trispheresegment           — hemisphere
%    6. trispherescale             — ellipsoid
%    7. tripolygon                 — hexagonal prism

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

data_dir = '/home/yoojk20/workspace/MNPBEM/validation/13_shapes/data';
fig_dir  = '/home/yoojk20/workspace/MNPBEM/validation/13_shapes/figures';

epstab = {epsconst(1), epstable('gold.dat')};
op = bemoptions('sim', 'stat', 'interp', 'curv');
enei = linspace(400, 800, 41);
n = length(enei);

timing = {};

%% ===== 1. trisphere =====
fprintf('=== 1. trisphere(144, 20) ===\n');
p_shape = trisphere(144, 20);
p = comparticle(epstab, {p_shape}, [2, 1], 1, op);
bem = bemsolver(p, op);
exc = planewave([1, 0, 0], [0, 0, 1], op);

ext1 = zeros(1, n);
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    ext1(i) = exc.extinction(sig);
end
t1 = toc;
fprintf('  Time: %.4f s, nfaces: %d\n', t1, size(p_shape.faces, 1));
timing{end+1, 1} = 'trisphere'; timing{end, 2} = t1;

fid = fopen(fullfile(data_dir, 'trisphere_matlab.csv'), 'w');
fprintf(fid, 'wavelength_nm,extinction\n');
for i = 1:n
    fprintf(fid, '%.6f,%.15e\n', enei(i), ext1(i));
end
fclose(fid);

%% ===== 2. trirod (x-pol + z-pol) =====
fprintf('=== 2. trirod(10, 40, [15,15,15]) ===\n');
p_shape = trirod(10, 40, [15, 15, 15]);
p = comparticle(epstab, {p_shape}, [2, 1], 1, op);
bem = bemsolver(p, op);

% x-polarization
exc_x = planewave([1, 0, 0], [0, 0, 1], op);
ext2x = zeros(1, n);
tic;
for i = 1:n
    sig = bem \ exc_x(p, enei(i));
    ext2x(i) = exc_x.extinction(sig);
end
t2x = toc;

% z-polarization
exc_z = planewave([0, 0, 1], [1, 0, 0], op);
ext2z = zeros(1, n);
tic;
for i = 1:n
    sig = bem \ exc_z(p, enei(i));
    ext2z(i) = exc_z.extinction(sig);
end
t2z = toc;

t2 = t2x + t2z;
fprintf('  Time: %.4f s (x: %.4f, z: %.4f), nfaces: %d\n', t2, t2x, t2z, size(p_shape.faces, 1));
timing{end+1, 1} = 'trirod'; timing{end, 2} = t2;

fid = fopen(fullfile(data_dir, 'trirod_matlab.csv'), 'w');
fprintf(fid, 'wavelength_nm,extinction_xpol,extinction_zpol\n');
for i = 1:n
    fprintf(fid, '%.6f,%.15e,%.15e\n', enei(i), ext2x(i), ext2z(i));
end
fclose(fid);

%% ===== 3. tricube =====
fprintf('=== 3. tricube(10, 20) ===\n');
p_shape = tricube(10, 20);
p = comparticle(epstab, {p_shape}, [2, 1], 1, op);
bem = bemsolver(p, op);
exc = planewave([1, 0, 0], [0, 0, 1], op);

ext3 = zeros(1, n);
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    ext3(i) = exc.extinction(sig);
end
t3 = toc;
fprintf('  Time: %.4f s, nfaces: %d\n', t3, size(p_shape.faces, 1));
timing{end+1, 1} = 'tricube'; timing{end, 2} = t3;

fid = fopen(fullfile(data_dir, 'tricube_matlab.csv'), 'w');
fprintf(fid, 'wavelength_nm,extinction\n');
for i = 1:n
    fprintf(fid, '%.6f,%.15e\n', enei(i), ext3(i));
end
fclose(fid);

%% ===== 4. tritorus =====
fprintf('=== 4. tritorus(15, 5, [20,20]) ===\n');
p_shape = tritorus(15, 5, [20, 20]);
p = comparticle(epstab, {p_shape}, [2, 1], 1, op);
bem = bemsolver(p, op);
exc = planewave([1, 0, 0], [0, 0, 1], op);

ext4 = zeros(1, n);
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    ext4(i) = exc.extinction(sig);
end
t4 = toc;
fprintf('  Time: %.4f s, nfaces: %d\n', t4, size(p_shape.faces, 1));
timing{end+1, 1} = 'tritorus'; timing{end, 2} = t4;

fid = fopen(fullfile(data_dir, 'tritorus_matlab.csv'), 'w');
fprintf(fid, 'wavelength_nm,extinction\n');
for i = 1:n
    fprintf(fid, '%.6f,%.15e\n', enei(i), ext4(i));
end
fclose(fid);

%% ===== 5. trispheresegment (hemisphere) =====
fprintf('=== 5. trispheresegment (hemisphere, d=20) ===\n');
phi = linspace(0, 2*pi, 15);
theta = linspace(0, pi/2, 10);
p_shape = trispheresegment(phi, theta, 20);
p = comparticle(epstab, {p_shape}, [2, 1], 1, op);
bem = bemsolver(p, op);
exc = planewave([1, 0, 0], [0, 0, 1], op);

ext5 = zeros(1, n);
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    ext5(i) = exc.extinction(sig);
end
t5 = toc;
fprintf('  Time: %.4f s, nfaces: %d\n', t5, size(p_shape.faces, 1));
timing{end+1, 1} = 'trispheresegment'; timing{end, 2} = t5;

fid = fopen(fullfile(data_dir, 'trispheresegment_matlab.csv'), 'w');
fprintf(fid, 'wavelength_nm,extinction\n');
for i = 1:n
    fprintf(fid, '%.6f,%.15e\n', enei(i), ext5(i));
end
fclose(fid);

%% ===== 6. trispherescale (ellipsoid) =====
fprintf('=== 6. trispherescale(trisphere(144,20), [1,1,2]) ===\n');
p_shape = scale(trisphere(144, 20), [1, 1, 2]);
p = comparticle(epstab, {p_shape}, [2, 1], 1, op);
bem = bemsolver(p, op);
exc = planewave([1, 0, 0], [0, 0, 1], op);

ext6 = zeros(1, n);
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    ext6(i) = exc.extinction(sig);
end
t6 = toc;
fprintf('  Time: %.4f s, nfaces: %d\n', t6, size(p_shape.faces, 1));
timing{end+1, 1} = 'trispherescale'; timing{end, 2} = t6;

fid = fopen(fullfile(data_dir, 'trispherescale_matlab.csv'), 'w');
fprintf(fid, 'wavelength_nm,extinction\n');
for i = 1:n
    fprintf(fid, '%.6f,%.15e\n', enei(i), ext6(i));
end
fclose(fid);

%% ===== 7. tripolygon (hexagonal prism) =====
fprintf('=== 7. tripolygon (hexagon + EdgeProfile) ===\n');
poly = polygon(6, 'size', [20, 20]);
edge = edgeprofile(5, 11);
p_shape = tripolygon(poly, edge);
p = comparticle(epstab, {p_shape}, [2, 1], 1, op);
bem = bemsolver(p, op);
exc = planewave([1, 0, 0], [0, 0, 1], op);

ext7 = zeros(1, n);
tic;
for i = 1:n
    sig = bem \ exc(p, enei(i));
    ext7(i) = exc.extinction(sig);
end
t7 = toc;
fprintf('  Time: %.4f s, nfaces: %d\n', t7, size(p_shape.faces, 1));
timing{end+1, 1} = 'tripolygon'; timing{end, 2} = t7;

fid = fopen(fullfile(data_dir, 'tripolygon_matlab.csv'), 'w');
fprintf(fid, 'wavelength_nm,extinction\n');
for i = 1:n
    fprintf(fid, '%.6f,%.15e\n', enei(i), ext7(i));
end
fclose(fid);

%% Save timing
fid = fopen(fullfile(data_dir, 'matlab_timing.csv'), 'w');
fprintf(fid, 'test,time_seconds\n');
for i = 1:size(timing, 1)
    fprintf(fid, '%s,%.6f\n', timing{i, 1}, timing{i, 2});
end
fclose(fid);

fprintf('\n=== Timing Summary ===\n');
total_time = 0;
for i = 1:size(timing, 1)
    fprintf('  %-20s : %.4f s\n', timing{i, 1}, timing{i, 2});
    total_time = total_time + timing{i, 2};
end
fprintf('  %-20s : %.4f s\n', 'TOTAL', total_time);

%% Plot individual MATLAB figures
shapes = {'trisphere', 'trirod', 'tricube', 'tritorus', ...
          'trispheresegment', 'trispherescale', 'tripolygon'};
ext_all = {ext1, [ext2x; ext2z], ext3, ext4, ext5, ext6, ext7};
titles_all = {'trisphere(144,20)', 'trirod(10,40,[15,15,15])', ...
              'tricube(10,20)', 'tritorus(15,5,[20,20])', ...
              'trispheresegment(d=20)', 'trispherescale([1,1,2])', ...
              'tripolygon(hex)'};

for s = 1:length(shapes)
    figure('Visible', 'off', 'Position', [100, 100, 800, 500]);
    if s == 2
        plot(enei, ext_all{s}(1,:), 'b-', 'LineWidth', 1.5); hold on;
        plot(enei, ext_all{s}(2,:), 'r--', 'LineWidth', 1.5);
        legend('x-pol', 'z-pol', 'Location', 'best');
    else
        plot(enei, ext_all{s}, 'b-', 'LineWidth', 1.5);
    end
    xlabel('Wavelength (nm)');
    ylabel('Extinction (nm^2)');
    title(sprintf('MATLAB BEMStat — %s', titles_all{s}));
    grid on;
    saveas(gcf, fullfile(fig_dir, [shapes{s} '_matlab.png']));
    close;
end

fprintf('\n[info] MATLAB 13_shapes validation complete.\n');
exit;
