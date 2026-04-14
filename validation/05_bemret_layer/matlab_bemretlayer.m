%% BEMRetLayer validation: 20nm Au sphere on glass substrate
%  Retarded BEM with layer structure + greentab
%  Normal incidence planewave, ext+sca vs 450-750nm (16 points)
%  Saves cross sections and timing to CSV

%% initialization
addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

% table of dielectric functions
epstab = { epsconst( 1 ), epstable( 'gold.dat' ), epsconst( 2.25 ) };

% layer structure: vacuum(1) above z=0, glass(3) below
layer = layerstructure( epstab, [ 1, 3 ], 0, layerstructure.options );

% BEM options
op = bemoptions( 'sim', 'ret', 'interp', 'curv', 'layer', layer );

% 20nm diameter gold sphere, 1nm above substrate
p = trisphere( 144, 20 );
p = shift( p, [ 0, 0, - min( p.pos( :, 3 ) ) + 1 ] );
p = comparticle( epstab, { p }, [ 2, 1 ], 1, op );

%% tabulated Green functions
tab = tabspace( layer, p );
greentab = compgreentablayer( layer, tab );
greentab = set( greentab, linspace( 350, 800, 5 ), op );
op.greentab = greentab;

%% BEM solver
bem = bemsolver( p, op );
exc = planewave( [ 1, 0, 0 ], [ 0, 0, -1 ], op );

% wavelength grid (16 points)
enei = linspace( 450, 750, 16 );

% preallocate
sca = zeros( numel( enei ), 1 );
ext = zeros( numel( enei ), 1 );

%% loop over wavelengths (with timing)
tic;
for ien = 1 : length( enei )
    sig = bem \ exc( p, enei( ien ) );
    sca( ien ) = exc.sca( sig );
    ext( ien ) = exc.extinction( sig );
    fprintf( '  [%d/%d] lambda = %.1f nm\n', ien, length( enei ), enei( ien ) );
end
elapsed = toc;

ab = ext - sca;

%% save results
output_dir = fullfile( fileparts( mfilename('fullpath') ), 'data' );
if ~exist( output_dir, 'dir' )
    mkdir( output_dir );
end

% cross sections
fid = fopen( fullfile( output_dir, 'matlab_retlayer.csv' ), 'w' );
fprintf( fid, 'wavelength_nm,scattering,extinction,absorption\n' );
for i = 1 : size( enei, 2 )
    fprintf( fid, '%.6f,%.10e,%.10e,%.10e\n', enei(i), sca(i), ext(i), ab(i) );
end
fclose( fid );

% timing
fid = fopen( fullfile( output_dir, 'matlab_retlayer_timing.csv' ), 'w' );
fprintf( fid, 'total_sec,n_wavelengths,per_wavelength_sec\n' );
fprintf( fid, '%.6f,%d,%.6f\n', elapsed, length( enei ), elapsed / length( enei ) );
fclose( fid );

fprintf( '\nMATLAB BEMRetLayer validation complete in %.1f sec.\n', elapsed );
fprintf( 'Results saved to %s\n', output_dir );
