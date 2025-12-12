"""
Step 2 Validation: Geometry and mesh generation

Test Particle, ComParticle, and trisphere against MATLAB MNPBEM.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from mnpbem.geometry import trisphere, Particle, ComParticle
from mnpbem.materials import EpsConst, EpsTable

print("=" * 70)
print("Step 2: Geometry and Mesh Generation Validation")
print("=" * 70)

# Test 1: trisphere - Generate sphere mesh
print("\n" + "-" * 70)
print("Test 1: trisphere - Sphere Mesh Generation")
print("-" * 70)

# Create 10nm sphere with ~144 vertices (as in MATLAB demospecstat1.m)
diameter = 10.0
sphere = trisphere(144, diameter)

print(f"\nCreated sphere:")
print(f"  Diameter: {diameter} nm")
print(f"  Vertices: {sphere.nverts}")
print(f"  Faces: {sphere.nfaces}")
print(f"  Total area: {sphere.area.sum():.4f} nm²")

# Theoretical surface area = 4πr² = πd²
theoretical_area = np.pi * diameter**2
print(f"  Theoretical area: {theoretical_area:.4f} nm²")
print(f"  Relative error: {abs(sphere.area.sum() - theoretical_area) / theoretical_area * 100:.2f}%")

# Check geometry properties
print(f"\nGeometric properties:")
print(f"  Centroids shape: {sphere.pos.shape}")
print(f"  Normal vectors shape: {sphere.nvec.shape}")
print(f"  Areas shape: {sphere.area.shape}")

# Verify normals are unit vectors
normal_lengths = np.linalg.norm(sphere.nvec, axis=1)
print(f"  Normal vector lengths (should be 1.0):")
print(f"    Mean: {normal_lengths.mean():.6f}")
print(f"    Std: {normal_lengths.std():.6e}")
print(f"    Min: {normal_lengths.min():.6f}, Max: {normal_lengths.max():.6f}")

# Verify normals point outward (for sphere, n·r > 0)
centroid_to_origin = sphere.pos / np.linalg.norm(sphere.pos, axis=1, keepdims=True)
dot_products = np.sum(sphere.nvec * centroid_to_origin, axis=1)
print(f"  Outward-pointing normals (n·r̂ should be > 0):")
print(f"    Mean: {dot_products.mean():.6f}")
print(f"    Min: {dot_products.min():.6f}, Max: {dot_products.max():.6f}")

# Test 2: Verify orthogonality of basis vectors
print("\n" + "-" * 70)
print("Test 2: Basis Vector Orthogonality")
print("-" * 70)

# vec1 ⊥ vec2
dot_v1_v2 = np.sum(sphere.vec1 * sphere.vec2, axis=1)
print(f"vec1 · vec2 (should be 0):")
print(f"  Mean: {dot_v1_v2.mean():.6e}")
print(f"  Max |dot|: {np.abs(dot_v1_v2).max():.6e}")

# vec1 ⊥ nvec
dot_v1_n = np.sum(sphere.vec1 * sphere.nvec, axis=1)
print(f"vec1 · nvec (should be 0):")
print(f"  Mean: {dot_v1_n.mean():.6e}")
print(f"  Max |dot|: {np.abs(dot_v1_n).max():.6e}")

# vec2 ⊥ nvec
dot_v2_n = np.sum(sphere.vec2 * sphere.nvec, axis=1)
print(f"vec2 · nvec (should be 0):")
print(f"  Mean: {dot_v2_n.mean():.6e}")
print(f"  Max |dot|: {np.abs(dot_v2_n).max():.6e}")

# Test 3: ComParticle - Compound particle with materials
print("\n" + "-" * 70)
print("Test 3: ComParticle - Gold Sphere in Vacuum")
print("-" * 70)

# Materials (as in demospecstat1.m)
eps_vacuum = EpsConst(1.0)
eps_gold = EpsTable('gold.dat')
epstab = [eps_vacuum, eps_gold]

# Create compound particle: gold sphere in vacuum
# inout = [2, 1] means inside=gold (index 2), outside=vacuum (index 1)
p = ComParticle(epstab, [sphere], [[2, 1]])

print(f"\n{p}")

# Test dielectric function evaluation
wavelength = 500.0
eps1 = p.eps1(wavelength)
eps2 = p.eps2(wavelength)

print(f"\nDielectric functions at λ = {wavelength} nm:")
print(f"  Inside (gold): ε = {eps1[0]:.4f}")
print(f"  Outside (vacuum): ε = {eps2[0]:.4f}")

# Verify geometry is preserved
print(f"\nGeometry consistency:")
print(f"  ComParticle total area: {p.area.sum():.4f} nm²")
print(f"  Sphere area: {sphere.area.sum():.4f} nm²")
print(f"  Match: {np.allclose(p.area.sum(), sphere.area.sum())}")

# Test 4: Multiple particle sizes
print("\n" + "-" * 70)
print("Test 4: Multiple Sphere Sizes")
print("-" * 70)

diameters = [5, 10, 20, 50]
print(f"\n{'Diameter (nm)':<15} {'Vertices':<12} {'Faces':<12} {'Area (nm²)':<15} {'Error (%)':<12}")
print("-" * 70)

for d in diameters:
    s = trisphere(144, d)
    theoretical = np.pi * d**2
    error = abs(s.area.sum() - theoretical) / theoretical * 100
    print(f"{d:<15.1f} {s.nverts:<12} {s.nfaces:<12} {s.area.sum():<15.2f} {error:<12.2f}")

# Test 5: Sphere at different resolutions
print("\n" + "-" * 70)
print("Test 5: Sphere Resolution Test")
print("-" * 70)

n_values = [50, 100, 200, 500]
diameter = 10.0
theoretical = np.pi * diameter**2

print(f"\nDiameter = {diameter} nm, Theoretical area = {theoretical:.4f} nm²")
print(f"\n{'N vertices':<15} {'Actual N':<12} {'Faces':<12} {'Area (nm²)':<15} {'Error (%)':<12}")
print("-" * 70)

for n in n_values:
    s = trisphere(n, diameter)
    error = abs(s.area.sum() - theoretical) / theoretical * 100
    print(f"{n:<15} {s.nverts:<12} {s.nfaces:<12} {s.area.sum():<15.4f} {error:<12.4f}")

# Visualization (if matplotlib available)
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(15, 5))

    # Plot 1: 3D sphere mesh
    ax1 = fig.add_subplot(131, projection='3d')
    sphere_test = trisphere(144, 10.0)

    # Plot vertices
    ax1.scatter(sphere_test.verts[:, 0],
                sphere_test.verts[:, 1],
                sphere_test.verts[:, 2],
                c='blue', s=10, alpha=0.6)

    # Plot a few faces
    for i in range(min(20, sphere_test.nfaces)):
        face_idx = sphere_test.faces[i, :3].astype(int)
        face_verts = sphere_test.verts[face_idx]
        face_verts = np.vstack([face_verts, face_verts[0]])
        ax1.plot(face_verts[:, 0], face_verts[:, 1], face_verts[:, 2], 'b-', alpha=0.3)

    ax1.set_xlabel('X (nm)')
    ax1.set_ylabel('Y (nm)')
    ax1.set_zlabel('Z (nm)')
    ax1.set_title('Sphere Mesh (10 nm)')
    ax1.set_box_aspect([1,1,1])

    # Plot 2: Area distribution
    ax2 = fig.add_subplot(132)
    ax2.hist(sphere_test.area, bins=30, edgecolor='black')
    ax2.set_xlabel('Face Area (nm²)')
    ax2.set_ylabel('Count')
    ax2.set_title('Face Area Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(sphere_test.area.mean(), color='r', linestyle='--',
                label=f'Mean: {sphere_test.area.mean():.4f} nm²')
    ax2.legend()

    # Plot 3: Normal vector check
    ax3 = fig.add_subplot(133)
    radii = np.linalg.norm(sphere_test.pos, axis=1)
    ax3.scatter(radii, dot_products, alpha=0.5, s=10)
    ax3.set_xlabel('Distance from origin (nm)')
    ax3.set_ylabel('n · r̂')
    ax3.set_title('Outward Normal Verification')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(1.0, color='r', linestyle='--', label='Expected (1.0)')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('/home/user/MNPBEM/mnpbem/examples/step2_geometry.png', dpi=150)
    print(f"\nPlot saved to: step2_geometry.png")

except ImportError:
    print("\nMatplotlib not available, skipping visualization")

print("\n" + "=" * 70)
print("Step 2 Validation Complete!")
print("=" * 70)
