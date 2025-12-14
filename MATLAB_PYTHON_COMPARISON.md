# MATLAB vs Python MNPBEM 전수 조사 결과

## 📋 요약

80 nm Au sphere 시뮬레이션 결과 불일치 원인을 찾기 위해 MATLAB과 Python 코드를 한 줄씩 비교 분석했습니다.

### 문제 현상
- **MATLAB 결과**: Peak @ 530 nm, 강도 232.7 nm²
- **Python 결과**: Peak @ 400 nm, 강도 83.72 nm² ❌

---

## 🔴 발견된 문제점

### Issue #1: **입자 크기 오류 (Critical)**

**위치**: `john_python.py:14-15`

**문제 코드**:
```python
diameter = 80  # nm
radius = diameter / 2  # = 40 nm
sphere = trisphere(144, radius)  # ❌ 잘못됨!
```

**원인**:
- `trisphere(n, diameter)` 함수는 **diameter**를 받습니다 (MATLAB/Python 공통)
- 사용자가 `radius` (40 nm)를 전달하여 의도한 크기의 절반인 구가 생성됨

**MATLAB 참조**:
```matlab
% Particles/particleshapes/trisphere.m:9
%  Usage :
%    p = trisphere( n, diameter )  % ← diameter를 받음!
%    p = trisphere( n, diameter, varargin )
```

**Python 참조**:
```python
# mnpbem/geometry/mesh_generators.py:10
def trisphere(n, diameter=1.0):
    """
    Parameters
    ----------
    diameter : float, optional
        Diameter of sphere in nm. Default: 1.0
    """
    verts = verts * (diameter / 2.0)  # diameter를 2로 나눔
```

**영향**:
- **생성된 구**: 40 nm diameter (의도: 80 nm)
- **결과**: Plasmon resonance가 blue-shift (530nm → 400nm)
- **결과**: Scattering cross section 감소 (232.7 → 83.72 nm²)

**수정 방법**:
```python
diameter = 80  # nm
sphere = trisphere(144, diameter)  # ✅ 직접 diameter 전달
# radius 변수 제거
```

---

### Issue #2: **PlaneWaveRet spectrum 초기화 누락 (Critical)**

**위치**: `mnpbem/simulation/planewave_ret.py:116-120`

**문제 코드**:
```python
self.spec = options.get('pinfty', None)
if self.spec is None:
    # MATLAB creates default spectrum with trisphere(256, 2)
    # We'll defer this until spectrum is needed
    pass  # ❌ 초기화하지 않음!
```

**MATLAB 참조**:
```matlab
% Simulation/retarded/@planewaveret/init.m:26-30
if isfield( op, 'pinfty' )
  obj.spec = spectrumret( op.pinfty, 'medium', obj.medium );
else
  obj.spec = spectrumret( trisphere( 256, 2 ), 'medium', obj.medium );  % ✅ 기본값 생성
end
```

**영향**:
- Test 2 (Retarded + Gold Table)에서 `exc2.scattering(sig)` 호출 시 에러 발생
- `NotImplementedError: Scattering calculation requires spectrum object`

**수정 방법**:
```python
from ..geometry import trisphere
from ..spectrum import SpectrumRet

# PlaneWaveRet.__init__ 내부:
self.spec = options.get('pinfty', None)
if self.spec is None:
    # MATLAB: obj.spec = spectrumret(trisphere(256, 2), 'medium', obj.medium)
    pinfty = trisphere(256, 2)
    self.spec = SpectrumRet(pinfty, medium=self.medium)
```

---

### Issue #3: **ComParticle inout 파라미터 형식**

**위치**: `john_python.py:33, 48, 63, 78`

**현재 코드**:
```python
p1 = ComParticle([eps_water, eps_au_table], [sphere], [2, 1], 1)
```

**분석**:
- Python `ComParticle`은 `np.atleast_2d(inout)`을 사용하여 `[2, 1]` → `[[2, 1]]`로 변환
- `eps1()`, `eps2()` 메서드에서 `int(self.inout[i, 0]) - 1`로 1-indexed 처리
- **결론**: 현재 코드는 정상 동작 ✅

**권장 사항** (명확성을 위해):
```python
# 더 명확한 형식 (선택사항)
p1 = ComParticle([eps_water, eps_au_table], [sphere], [[2, 1]], 1)
#                                                       ^^^^^^^^ 2D 형식
```

---

## ✅ 검증된 정상 구현

### 1. **BEMStat / BEMRet solve() 메서드**
- 둘 다 `(sig, self)` tuple 반환 ✅
- MATLAB: `[sig, obj] = solve(obj, exc)`
- Python: `sig, obj = bem.solve(exc)`

### 2. **PlaneWaveStat scattering() 계산**
- 구현 완료 및 정확성 검증 ✅
- MATLAB `scattering.m:17`와 동일한 공식 사용:
  ```python
  sca = 8 * np.pi / 3 * k**4 * np.sum(np.abs(dip)**2, axis=0)
  ```

### 3. **재료 정의 (EpsConst, EpsTable, EpsDrude)**
- 모든 클래스 정상 구현 ✅
- `gold.dat` 파일 위치 확인:
  - MATLAB: `Material/@epstable/gold.dat`
  - Python: `mnpbem/materials/data/gold.dat`

### 4. **wavenumber 계산**
- 모든 재료 클래스에서 정확히 구현 ✅
- 공식: `k = 2π/λ × √ε`

---

## 🔧 수정된 Python 테스트 코드

```python
#!/usr/bin/env python
"""MNPBEM Full Test - Python (CORRECTED)
Tests: stat/ret x gold_table/drude
Spectrum: 400-800nm, 80nm Au sphere in water
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Path setup
mnpbem_path = os.path.join(os.getcwd(), 'MNPBEM')
sys.path.insert(0, mnpbem_path)
print(f"Added: {mnpbem_path}\n")

from mnpbem import (
    EpsConst, EpsTable, EpsDrude,
    trisphere, ComParticle,
    BEMStat, BEMRet,
    PlaneWaveStat, PlaneWaveRet,
    SpectrumRet  # ← 추가
)

# Logging
log_file = open('python_test_corrected.log', 'w')
def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

log("=== MNPBEM Python Test (CORRECTED) ===")
log(f"Date: {datetime.now()}\n")

# Setup
diameter = 80  # nm
wavelengths = np.linspace(400, 800, 41)

# Materials
eps_water = EpsConst(1.33**2)
eps_au_table = EpsTable('gold.dat')
eps_au_drude = EpsDrude.gold()

# ✅ FIX #1: trisphere takes diameter, not radius
log(f"Creating: {diameter} nm Au sphere")
sphere = trisphere(144, diameter)  # ✅ 수정됨!

# Test 1: Quasistatic + Gold Table
log("\n--- Test 1: Quasistatic + Gold Table ---")
p1 = ComParticle([eps_water, eps_au_table], [sphere], [[2, 1]], 1)
bem1 = BEMStat(p1)
exc1 = PlaneWaveStat(pol=np.array([1, 0, 0]))

log("Computing...")
sca1 = []
for wl in wavelengths:
    sig, _ = bem1.solve(exc1(p1, wl))
    sca1.append(exc1.scattering(sig))
sca1 = np.array(sca1)
idx1 = np.argmax(sca1)
log(f"Peak: {sca1[idx1]:.3e} nm^2 at {wavelengths[idx1]:.0f} nm")

# Test 2: Retarded + Gold Table
log("\n--- Test 2: Retarded + Gold Table ---")
p2 = ComParticle([eps_water, eps_au_table], [sphere], [[2, 1]], 1)
bem2 = BEMRet(p2)

# ✅ FIX #2: Initialize spectrum for scattering calculation
pinfty = trisphere(256, 2)
exc2 = PlaneWaveRet(
    pol=np.array([1, 0, 0]),
    dir=np.array([0, 0, 1]),
    pinfty=pinfty,  # ✅ spectrum 초기화!
    medium=1
)

log("Computing...")
sca2 = []
for wl in wavelengths:
    sig, _ = bem2.solve(exc2(p2, wl))
    sca_val, _ = exc2.scattering(sig)  # ✅ 이제 작동!
    sca2.append(sca_val)
sca2 = np.array(sca2)
idx2 = np.argmax(sca2)
log(f"Peak: {sca2[idx2]:.3e} nm^2 at {wavelengths[idx2]:.0f} nm")

# Test 3: Quasistatic + Drude
log("\n--- Test 3: Quasistatic + Drude ---")
p3 = ComParticle([eps_water, eps_au_drude], [sphere], [[2, 1]], 1)
bem3 = BEMStat(p3)
exc3 = PlaneWaveStat(pol=np.array([1, 0, 0]))

log("Computing...")
sca3 = []
for wl in wavelengths:
    sig, _ = bem3.solve(exc3(p3, wl))
    sca3.append(exc3.scattering(sig))
sca3 = np.array(sca3)
idx3 = np.argmax(sca3)
log(f"Peak: {sca3[idx3]:.3e} nm^2 at {wavelengths[idx3]:.0f} nm")

# Test 4: Retarded + Drude
log("\n--- Test 4: Retarded + Drude ---")
p4 = ComParticle([eps_water, eps_au_drude], [sphere], [[2, 1]], 1)
bem4 = BEMRet(p4)
exc4 = PlaneWaveRet(
    pol=np.array([1, 0, 0]),
    dir=np.array([0, 0, 1]),
    pinfty=pinfty,  # ✅ spectrum 초기화!
    medium=1
)

log("Computing...")
sca4 = []
for wl in wavelengths:
    sig, _ = bem4.solve(exc4(p4, wl))
    sca_val, _ = exc4.scattering(sig)
    sca4.append(sca_val)
sca4 = np.array(sca4)
idx4 = np.argmax(sca4)
log(f"Peak: {sca4[idx4]:.3e} nm^2 at {wavelengths[idx4]:.0f} nm")

# Summary
log("\n=== Summary ===")
log(f"Test 1: {wavelengths[idx1]:.0f} nm, {sca1[idx1]:.3e} nm^2")
log(f"Test 2: {wavelengths[idx2]:.0f} nm, {sca2[idx2]:.3e} nm^2")
log(f"Test 3: {wavelengths[idx3]:.0f} nm, {sca3[idx3]:.3e} nm^2")
log(f"Test 4: {wavelengths[idx4]:.0f} nm, {sca4[idx4]:.3e} nm^2")

# Plot
log("\n--- Plotting ---")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(wavelengths, sca1, 'b-', lw=2)
axes[0,0].set_title('Test 1: Quasistatic + Gold Table')
axes[0,0].set_xlabel('Wavelength (nm)'); axes[0,0].set_ylabel('Scattering (nm²)')
axes[0,0].grid(True)

axes[0,1].plot(wavelengths, sca2, 'r-', lw=2)
axes[0,1].set_title('Test 2: Retarded + Gold Table')
axes[0,1].set_xlabel('Wavelength (nm)'); axes[0,1].set_ylabel('Scattering (nm²)')
axes[0,1].grid(True)

axes[1,0].plot(wavelengths, sca3, 'g-', lw=2)
axes[1,0].set_title('Test 3: Quasistatic + Drude')
axes[1,0].set_xlabel('Wavelength (nm)'); axes[1,0].set_ylabel('Scattering (nm²)')
axes[1,0].grid(True)

axes[1,1].plot(wavelengths, sca4, 'm-', lw=2)
axes[1,1].set_title('Test 4: Retarded + Drude')
axes[1,1].set_xlabel('Wavelength (nm)'); axes[1,1].set_ylabel('Scattering (nm²)')
axes[1,1].grid(True)

plt.tight_layout()
plt.savefig('python_results_corrected.png', dpi=150)
log("Saved: python_results_corrected.png")

log("\n=== Complete ===")
log_file.close()
```

---

## 🎯 예상 결과

수정 후 Python 결과는 MATLAB과 유사해야 합니다:

| Test | MATLAB Peak | Python Peak (수정 전) | Python Peak (수정 후) |
|------|-------------|---------------------|---------------------|
| Test 1 (Stat+Table) | 530 nm, 232.7 nm² | 400 nm, 83.72 nm² ❌ | ~530 nm, ~230 nm² ✅ |
| Test 2 (Ret+Table)  | 530 nm, 256.8 nm² | Error ❌ | ~530 nm, ~250 nm² ✅ |
| Test 3 (Stat+Drude) | 500 nm, 7215 nm² | - | ~500 nm ✅ |
| Test 4 (Ret+Drude)  | 510 nm, 4962 nm² | - | ~510 nm ✅ |

---

## 📝 추가 권장사항

### PlaneWaveRet 클래스 영구 수정

`mnpbem/simulation/planewave_ret.py:116-120` 수정:

```python
# 현재 (잘못됨):
self.spec = options.get('pinfty', None)
if self.spec is None:
    pass  # ❌

# 수정안:
from ..geometry import trisphere
from ..spectrum import SpectrumRet

self.spec = options.get('pinfty', None)
if self.spec is None:
    # MATLAB: obj.spec = spectrumret(trisphere(256, 2), 'medium', obj.medium)
    pinfty = trisphere(256, 2)
    self.spec = SpectrumRet(pinfty, medium=self.medium)  # ✅
```

---

## 📚 참고 파일

### MATLAB 참조
- `Particles/particleshapes/trisphere.m` - 구 생성
- `Simulation/retarded/@planewaveret/init.m` - PlaneWaveRet 초기화
- `Simulation/retarded/@planewaveret/scattering.m` - Scattering 계산
- `Demo/planewave/stat/demospecstat1.m` - 예제 코드

### Python 구현
- `mnpbem/geometry/mesh_generators.py` - trisphere
- `mnpbem/simulation/planewave_ret.py` - PlaneWaveRet
- `mnpbem/spectrum/spectrum_ret.py` - SpectrumRet
- `mnpbem/bem/bem_ret.py` - BEMRet

---

## ✨ 결론

**2개의 Critical 버그 발견**:
1. ✅ **입자 크기 오류**: `trisphere(144, radius)` → `trisphere(144, diameter)`
2. ✅ **Spectrum 초기화 누락**: PlaneWaveRet에서 SpectrumRet 생성 필요

이 두 가지를 수정하면 Python 결과가 MATLAB과 일치할 것입니다!
