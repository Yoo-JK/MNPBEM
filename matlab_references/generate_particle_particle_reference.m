% Generate reference data for particle.particle()
%
% This script runs the MATLAB version and saves outputs for Python comparison

function generate_particle_particle_reference()
    % Setup
    addpath(genpath('../'));

    % Create test inputs (example - adjust based on actual requirements)
    % obj = particle(...);

    % Execute method
    % result = particle(obj, ...);

    % Save reference data
    save('../../tests/references/particle_particle_ref.mat', ...
         'result', '-v7.3');

    fprintf('Reference data saved to particle_particle_ref.mat\n');
end