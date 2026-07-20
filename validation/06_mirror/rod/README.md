# 06_mirror / rod — Skipped

Unlike `trispheresegment`, MNPBEM does not naturally provide a builder that generates only the symmetric region (1/4) for the `trirod` geometry. Validating a rod mirror with `comparticlemirror` + `sym = 'xy'` would require the additional work of cutting the trirod mesh to 1/4, so this validation case is applied only to the sphere.

If you want mirror symmetry validation for the rod as well, it can be extended in one of the following ways:
- Write a quarter-only mesh generation helper as a variant of the `trirod` implementation
- Use an alternative `tripolygon`-based geometry in the style of the MATLAB demo `demospecret16.m` (nanodisk + mirror)
