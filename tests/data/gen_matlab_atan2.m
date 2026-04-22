% Generate MATLAB atan2 reference values for matan2 bit-identical tests.
% Produces tests/data/matlab_atan2.bin: [y(N); x(N); r(N)] raw float64.
rng(42);
N = 20000;
y_rand = (rand(N,1)*2 - 1) * 100;
x_rand = (rand(N,1)*2 - 1) * 100;
y_small = (rand(N,1)*2 - 1) * 1e-3;
x_small = (rand(N,1)*2 - 1) * 1e-3;
y_axis = [zeros(100,1); rand(100,1); -rand(100,1)];
x_axis = [rand(100,1); zeros(100,1); zeros(100,1)];
y = [y_rand; y_small; y_axis; 0; 1; -1; 0; 0; 1; -1; 1; -1];
x = [x_rand; x_small; x_axis; 1; 0; 0; -1; 0; 1; 1; -1; -1];
r = atan2(y, x);
fid = fopen('matlab_atan2.bin','w');
fwrite(fid, y, 'double');
fwrite(fid, x, 'double');
fwrite(fid, r, 'double');
fclose(fid);
fprintf('wrote %d samples\n', length(y));
