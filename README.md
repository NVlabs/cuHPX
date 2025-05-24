# cuHPX - GPU-Accelerated utilities for data on HEALPix grids

cuHPX is a library for performing transformations and analysis on HEALPix using
the GPU. Currently, it supports data shuffling between the RING/NESTED layout
and the flat index layout used in the earth-2 grid. Other features include
differentiable spherical harmonic transformations, and regriding and
interpolation between HEALPix and equiangular lat/lon grids.

To setup the library, run

```bash
git clone https://gitlab-master.nvidia.com/Devtech-Compute/cuhpx.git
cd cuhpx
python3 setup.py build
pip install .
```

Example with cuhpx for real spherical harmonic transformations and its inverse
on HEALPix:

```python
import torch
from cuhpx import SHTCUDA, iSHTCUDA

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nside = 32
npix = 12* nside**2
signal = torch.randn(npix, dtype = torch.float32).to(device)

quad_weights = 'ring'
lmax = 2*nside+1
mmax = lmax

sht = SHTCUDA(nside, lmax = lmax, mmax = mmax, quad_weights = quad_weights)
isht = iSHTCUDA(nside, lmax = lmax, mmax = mmax)

coeff = sht(signal)
signal_back = isht(coeff)
```

Example with cuhpx for regridding between HEALPix and equiangular grid:

```python
import torch, cuhpx
from cuhpx import Grid, Regridding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nside = 32
nlat, nlon = 64, 128
lmax, mmax = 65, 65
signal_hpx = torch.randn(12*nside**2, dtype=torch.float32).to(device)

src_grid = Grid('healpix', nside)
dest_grid = Grid('equiangular', (nlat, nlon))

hpx2eq = Regridding(src_grid, dest_grid, lmax=lmax, mmax=mmax, device=device)
eq2hpx = Regridding(dest_grid, src_grid, lmax=lmax, mmax=mmax, device=device)

signal_eq = hpx2eq.execute(signal_hpx)
signal_back = eq2hpx.execute(signal_eq)
```

Example with cuhpx for remapping from NESTED to flat layout in earth2-grid:

```python
import torch
import cuhpx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nside = 2**10
npix = 12*nside**2
nbatch = 64

tensor_src = torch.randn(nbatch, npix, device=device, dtype=torch.float32)
origin = 'E' # origin can be E, S, W, N
clockwise = False # False for Counterclockwise; True for clockwise

tensor_dest = cuhpx.nest2flat(tensor_src, origin, clockwise, nside)

result_tensor = tensor_dest.to('cpu')
```

Note that for functions involving the flat/xy layout for earth2-grid, the
origin and clockwise must be declared.

Example for remapping from flat layout in earth2-grid to RING:

```python
tensor_in_ring = cuhpx.flat2ring(tensor_in_xy, origin, clockwise, nside)
```

Example for remapping from flat layout to flat layout but with different
origin or clockwise:

```python
tensor_dest = cuhpx.flat2flat(tensor_src, src_origin, src_clockwise,
                              dest_origin, dest_clockwise, nside)
```

Example for remapping between RING and NESTED:

```python
tensor_in_nest = cuhpx.ring2nest(tensor_in_ring, nside)
```

Included functions: `nest2flat()`, `ring2flat()`, `flat2nest()`,
`flat2ring()`, `flat2flat()`, `nest2ring()`, `ring2nest()`,

For processing data remapping in batch: `nest2flat_batch()`,
`ring2flat_batch()`, `flat2nest_batch()`, `flat2ring_batch()`,
`flat2flat_batch()`, `nest2ring_batch()`, `ring2nest_batch()`.
The input should be a 3D tensor.

For verification:

```bash
python3 test/data_remapping_test.py
python3 test/harmonic_transform_test.py
python3 test/differentiability_test.py
python3 test/regridding_test.py
```
