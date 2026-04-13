%% Sphere on substrate - retarded BEM comparison
%  Gold nanosphere (diameter 20nm) on glass substrate (eps=2.25)
%  Normal incidence planewave, TM polarization
%  Save scattering, extinction, absorption cross sections to CSV

%% initialization
addpath(genpath('/home/yoojk20/workspace/MNPBEM'));

% table of dielectric functions
epstab = { epsconst( 1 ), epstable( 'gold.dat' ), epsconst( 2.25 ) };
% location of interface of substrate
ztab = 0;

% default options for layer structure
op = layerstructure.options;
% set up layer structure
layer = layerstructure( epstab, [ 1, 3 ], ztab, op );
% options for BEM simulations (retarded only)
op2 = bemoptions( 'sim', 'ret', 'interp', 'curv', 'layer', layer );

% initialize nanosphere (20nm diameter, ~144 faces)
p = trisphere( 144, 20 );
% shift nanosphere 1 nm above layer
p = shift( p, [ 0, 0, - min( p.pos( :, 3 ) ) + 1 + ztab ] );

% set up COMPARTICLE object: gold(2) inside, vacuum(1) outside
p = comparticle( epstab, { p }, [ 2, 1 ], 1, op2 );

% single propagation angle: normal incidence from above
pol = [ 1, 0, 0 ];
dir = [ 0, 0, -1 ];

% photon wavelength range (nm)
enei = linspace( 450, 750, 31 );

%% tabulated Green functions
tab = tabspace( layer, p );
greentab = compgreentablayer( layer, tab );
greentab = set( greentab, linspace( 350, 800, 5 ), op2 );
op2.greentab = greentab;

%% BEM solver
bem = bemsolver( p, op2 );
exc = planewave( pol, dir, op2 );

% preallocate
sca = zeros( numel( enei ), 1 );
ext = zeros( numel( enei ), 1 );

%% loop over wavelengths
for ien = 1 : length( enei )
    sig = bem \ exc( p, enei( ien ) );
    sca( ien ) = exc.sca( sig );
    ext( ien ) = exc.extinction( sig );
end

% compute absorption = extinction - scattering
ab = ext - sca;

%% save results
output_dir = fileparts( mfilename('fullpath') );
results = [ enei', sca, ext, ab ];

% write header and data
fid = fopen( fullfile( output_dir, 'matlab_results.csv' ), 'w' );
fprintf( fid, 'wavelength_nm,scattering,extinction,absorption\n' );
for i = 1 : size( results, 1 )
    fprintf( fid, '%.6f,%.10e,%.10e,%.10e\n', results(i,1), results(i,2), results(i,3), results(i,4) );
end
fclose( fid );

fprintf('MATLAB simulation complete. Results saved to comparison/matlab_results.csv\n');
fprintf('Wavelength range: %.0f - %.0f nm (%d points)\n', enei(1), enei(end), numel(enei));
[~, idx] = max(sca);
fprintf('Peak scattering at %.1f nm: %.4e nm^2\n', enei(idx), max(sca));
