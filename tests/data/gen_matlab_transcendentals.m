% Generate MATLAB reference values for all bit-identical transcendentals.
% Produces one .bin per function: raw float64 [x(N); r(N)] for unary or
% [a(N); b(N); r(N)] for binary.
%
% Domain rules (important): MATLAB auto-promotes to complex on negative
% inputs for log/sqrt/asin/acos/log1p/log2/log10/power. The vector C-API
% (mu::Log1p<double>, ...) is real-only and returns NaN via error count.
% So we restrict inputs to the real domain where the scalar double output
% is well-defined, matching what the Python wrappers compute.
rng(42);
N = 4000;

x_std   = (rand(N,1)*2 - 1) * 100;
x_small = (rand(N,1)*2 - 1) * 1e-3;
x_pos   = rand(N,1) * 100 + 1e-6;       % strictly positive for log/sqrt
x_unit  = rand(N,1)*2 - 1;              % [-1, 1] for asin/acos
x_angle = (rand(N,1)*2 - 1) * pi;
x_hyp   = (rand(N,1)*2 - 1) * 5;
x_edges = [0; 1; -1; 0.5; -0.5; 2; -2; 10; -10; 100; -100; ...
           1e-300; -1e-300; 1e300; -1e300; eps; -eps; pi; -pi; pi/2; -pi/2];
x_pos_edges = x_edges(x_edges > 0);
x_nonneg_edges = x_edges(x_edges >= 0);
% log1p domain: x > -1
x_log1p = [x_small; rand(N,1)*200; x_pos_edges; 0];

write_unary('matlab_exp.bin',   [x_std; x_small; x_edges], @exp);
write_unary('matlab_log.bin',   [x_pos; x_pos_edges], @log);
write_unary('matlab_log10.bin', [x_pos; x_pos_edges], @log10);
write_unary('matlab_log2.bin',  [x_pos; x_pos_edges], @log2);
write_unary('matlab_log1p.bin', x_log1p, @log1p);
write_unary('matlab_expm1.bin', [x_std; x_small; x_edges], @expm1);
write_unary('matlab_sqrt.bin',  [x_pos; x_nonneg_edges], @sqrt);
write_unary('matlab_sin.bin',   [x_angle; x_std; x_edges], @sin);
write_unary('matlab_cos.bin',   [x_angle; x_std; x_edges], @cos);
write_unary('matlab_tan.bin',   [x_angle; x_small; x_edges], @tan);
write_unary('matlab_sinh.bin',  [x_hyp; x_small; x_edges], @sinh);
write_unary('matlab_cosh.bin',  [x_hyp; x_small; x_edges], @cosh);
write_unary('matlab_tanh.bin',  [x_hyp; x_std; x_edges], @tanh);
write_unary('matlab_asin.bin',  [x_unit; [0; 1; -1; 0.5; -0.5]], @asin);
write_unary('matlab_acos.bin',  [x_unit; [0; 1; -1; 0.5; -0.5]], @acos);
write_unary('matlab_atan.bin',  [x_std; x_small; x_edges], @atan);
write_unary('matlab_abs.bin',   [x_std; x_small; x_edges], @abs);
write_unary('matlab_sign.bin',  [x_std; x_small; x_edges], @sign);
write_unary('matlab_round.bin', [x_std; [0.5; -0.5; 1.5; -1.5; 2.5; -2.5; 0.49999999999; -0.49999999999]], @round);
write_unary('matlab_floor.bin', [x_std; x_edges], @floor);
write_unary('matlab_ceil.bin',  [x_std; x_edges], @ceil);
write_unary('matlab_fix.bin',   [x_std; x_edges], @fix);

a_std = (rand(N,1)*2 - 1) * 50;
b_std = (rand(N,1)*2 - 1) * 50;
% power: restrict base > 0 to stay real-valued
pw_base = rand(N,1) * 10 + 0.1;
pw_exp  = (rand(N,1)*2 - 1) * 5;
write_binary('matlab_power.bin', [pw_base; [2;3;10;0.5]], [pw_exp; [10;3;2;4]], @power);
write_binary('matlab_hypot.bin', [a_std; [3;0;1;1e300]], [b_std; [4;0;1;1e300]], @hypot);

fprintf('done\n');

function write_unary(fname, x, fn)
    r = fn(x);
    assert(isreal(r), 'non-real output for %s', fname);
    fid = fopen(fname, 'w');
    fwrite(fid, x, 'double');
    fwrite(fid, r, 'double');
    fclose(fid);
    fprintf('%s: %d samples\n', fname, length(x));
end

function write_binary(fname, a, b, fn)
    if length(a) ~= length(b)
        n = min(length(a), length(b));
        a = a(1:n); b = b(1:n);
    end
    r = fn(a, b);
    assert(isreal(r), 'non-real output for %s', fname);
    fid = fopen(fname, 'w');
    fwrite(fid, a, 'double');
    fwrite(fid, b, 'double');
    fwrite(fid, r, 'double');
    fclose(fid);
    fprintf('%s: %d samples\n', fname, length(a));
end
