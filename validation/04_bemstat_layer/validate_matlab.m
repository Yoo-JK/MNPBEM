%% BEMStatLayer validation: Au sphere on glass substrate (quasistatic)
%  20nm Au sphere, 1nm gap above z=0 substrate (eps=2.25)
%  Two angles: normal incidence (theta=0) and oblique (theta=45, TM)
%  Extinction and scattering vs wavelength 400-800nm (41 points)

addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

%% Materials and layer
epstab = { epsconst( 1 ), epstable( 'gold.dat' ), epsconst( 2.25 ) };
ztab = 0;

op = layerstructure.options;
layer = layerstructure( epstab, [ 1, 3 ], ztab, op );
op = bemoptions( 'sim', 'stat', 'interp', 'curv', 'layer', layer );

%% Particle: 20nm Au sphere, 1nm above substrate
p = trisphere( 144, 20 );
p = shift( p, [ 0, 0, - min( p.pos( :, 3 ) ) + 1 ] );
p = comparticle( epstab, { p }, [ 2, 1 ], 1, op );

%% Wavelength grid
enei = linspace( 400, 800, 41 );

%% Output directory
output_dir = fullfile( fileparts( mfilename('fullpath') ), 'data' );

%% ==============================
%%  Case 1: Normal incidence
%% ==============================
fprintf('=== Normal incidence (theta=0) ===\n');

pol_n = [ 1, 0, 0 ];
dir_n = [ 0, 0, -1 ];

bem = bemsolver( p, op );
exc = planewave( pol_n, dir_n, op );

sca_n = zeros( numel( enei ), 1 );
ext_n = zeros( numel( enei ), 1 );

tic;
for ien = 1 : length( enei )
    sig = bem \ exc( p, enei( ien ) );
    sca_n( ien ) = exc.sca( sig );
    ext_n( ien ) = exc.extinction( sig );
    fprintf( '  [%d/%d] lambda = %.1f nm\n', ien, length( enei ), enei( ien ) );
end
time_normal = toc;
fprintf( 'Normal incidence done in %.2f sec\n\n', time_normal );

% Save normal incidence results
fid = fopen( fullfile( output_dir, 'matlab_normal.csv' ), 'w' );
fprintf( fid, 'wavelength_nm,extinction,scattering\n' );
for i = 1 : numel( enei )
    fprintf( fid, '%.6f,%.10e,%.10e\n', enei(i), ext_n(i), sca_n(i) );
end
fclose( fid );

%% ==============================
%%  Case 2: Oblique incidence (theta=45, TM)
%% ==============================
fprintf('=== Oblique incidence (theta=45, TM) ===\n');

theta = pi / 4;
pol_o = [ cos( theta ), 0, sin( theta ) ];
dir_o = [ sin( theta ), 0, -cos( theta ) ];

exc2 = planewave( pol_o, dir_o, op );

sca_o = zeros( numel( enei ), 1 );
ext_o = zeros( numel( enei ), 1 );

tic;
for ien = 1 : length( enei )
    sig = bem \ exc2( p, enei( ien ) );
    sca_o( ien ) = exc2.sca( sig );
    ext_o( ien ) = exc2.extinction( sig );
    fprintf( '  [%d/%d] lambda = %.1f nm\n', ien, length( enei ), enei( ien ) );
end
time_oblique = toc;
fprintf( 'Oblique incidence done in %.2f sec\n\n', time_oblique );

% Save oblique incidence results
fid = fopen( fullfile( output_dir, 'matlab_oblique.csv' ), 'w' );
fprintf( fid, 'wavelength_nm,extinction,scattering\n' );
for i = 1 : numel( enei )
    fprintf( fid, '%.6f,%.10e,%.10e\n', enei(i), ext_o(i), sca_o(i) );
end
fclose( fid );

% Save timing
fid = fopen( fullfile( output_dir, 'matlab_timing.csv' ), 'w' );
fprintf( fid, 'case,time_sec\n' );
fprintf( fid, 'normal,%.4f\n', time_normal );
fprintf( fid, 'oblique,%.4f\n', time_oblique );
fprintf( fid, 'total,%.4f\n', time_normal + time_oblique );
fclose( fid );

%% Summary
fprintf( '============================================================\n' );
fprintf( 'MATLAB BEMStatLayer validation complete\n' );
fprintf( '  Normal:  %.2f sec\n', time_normal );
fprintf( '  Oblique: %.2f sec\n', time_oblique );
fprintf( '  Total:   %.2f sec\n', time_normal + time_oblique );
fprintf( '============================================================\n' );

[~, idx_n] = max( ext_n );
fprintf( 'Normal  - peak ext at %.1f nm: %.4e nm^2\n', enei( idx_n ), ext_n( idx_n ) );
[~, idx_o] = max( ext_o );
fprintf( 'Oblique - peak ext at %.1f nm: %.4e nm^2\n', enei( idx_o ), ext_o( idx_o ) );

%% Plots
fig1 = figure('Visible', 'off');
subplot(1,2,1);
plot( enei, ext_n, 'b-o', 'MarkerSize', 3 ); hold on;
plot( enei, sca_n, 'r-s', 'MarkerSize', 3 );
xlabel( 'Wavelength (nm)' );
ylabel( 'Cross section (nm^2)' );
title( 'Normal incidence (\theta=0)' );
legend( 'Extinction', 'Scattering' );
grid on;

subplot(1,2,2);
plot( enei, ext_o, 'b-o', 'MarkerSize', 3 ); hold on;
plot( enei, sca_o, 'r-s', 'MarkerSize', 3 );
xlabel( 'Wavelength (nm)' );
ylabel( 'Cross section (nm^2)' );
title( 'Oblique incidence (\theta=45, TM)' );
legend( 'Extinction', 'Scattering' );
grid on;

sgtitle( 'MATLAB BEMStatLayer: Au sphere on glass' );

fig_dir = fullfile( fileparts( mfilename('fullpath') ), 'figures' );
saveas( fig1, fullfile( fig_dir, 'matlab_normal.png' ) );

fig2 = figure('Visible', 'off');
subplot(1,2,1);
plot( enei, ext_n, 'b-o', 'MarkerSize', 3 ); hold on;
plot( enei, sca_n, 'r-s', 'MarkerSize', 3 );
xlabel( 'Wavelength (nm)' );
ylabel( 'Cross section (nm^2)' );
title( 'Normal incidence' );
legend( 'Extinction', 'Scattering' );
grid on;
saveas( fig2, fullfile( fig_dir, 'matlab_normal.png' ) );

fig3 = figure('Visible', 'off');
plot( enei, ext_o, 'b-o', 'MarkerSize', 3 ); hold on;
plot( enei, sca_o, 'r-s', 'MarkerSize', 3 );
xlabel( 'Wavelength (nm)' );
ylabel( 'Cross section (nm^2)' );
title( 'Oblique incidence (\theta=45, TM)' );
legend( 'Extinction', 'Scattering' );
grid on;
saveas( fig3, fullfile( fig_dir, 'matlab_oblique.png' ) );

fprintf( 'Figures saved.\n' );
exit;
