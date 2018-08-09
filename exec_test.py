import matplotlib.pyplot as plt
import numpy as np
import time

np.random.seed(int(time.time()))

plt.ion()
plt.close('all')

grid = plt.GridSpec(2, 4, wspace=1.0, hspace=0.4)
fig = plt.figure(figsize=(9, 4))
x, y = 0.07, 0.77

# The PDFs of the individual pixels.
ax = plt.subplot(grid[0, 1:3])
ax.plot([0, 0.05, 0.05, 0.15, 0.15, 0.25, 0.25, 0.35,
         0.35, 1.00],
        [0, 0.00, 0.10, 0.10, 0.80, 0.80, 0.10, 0.10,
         0.00, 0.00],
        label='Pixel A')
ax.plot([0, 0.65, 0.65, 0.75, 0.75, 0.85, 0.85, 0.95,
         0.95, 1.00],
        [0, 0.00, 0.10, 0.10, 0.80, 0.80, 0.10, 0.10,
         0.00, 0.00],
        label='Pixel B')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel('DGR')
ax.set_ylabel('P(DGR)')
ax.set_yticks([])
ax.legend()
ax.set_title('(a)', x=x, y=y)

ax = plt.subplot(grid[1, 0:2])
n = 100000
SigmaD_randA = np.random.random(n)
SigmaD_A = np.empty_like(SigmaD_randA)
SigmaD_A[SigmaD_randA < 0.1] = 0.1
SigmaD_A[(0.1 <= SigmaD_randA) * (SigmaD_randA < 0.9)] = 0.2
SigmaD_A[0.9 <= SigmaD_randA] = 0.3
SigmaD_randB = np.random.random(n)
SigmaD_B = np.empty_like(SigmaD_randB)
SigmaD_B[SigmaD_randB < 0.1] = 0.7
SigmaD_B[(0.1 <= SigmaD_randB) * (SigmaD_randB < 0.9)] = 0.8
SigmaD_B[0.9 <= SigmaD_randB] = 0.9
DGRs = (SigmaD_A + SigmaD_B) / 2
n, bins, patches = ax.hist(DGRs, bins=np.arange(21) / 20 - 0.025, density=True,
                           label='Realize')
ylim = [0.0, np.nanmax(n)/0.8]
bins2 = \
    [0.425, 0.475, 0.525, 0.575]
csp = \
    [0.010, 0.170, 0.830, 0.990]
DGR16, DGR84 = np.interp([0.16, 0.84], csp, bins2)
ax.plot([DGR16] * 2, ylim, label='16%')
ax.plot([DGR84] * 2, ylim, label='84%')
ax.set_xlim([0.0, 1.0])
ax.set_ylim(ylim)
ax.set_xlabel('DGR')
ax.set_ylabel('P(DGR)')
ax.set_yticks([])
ax.legend()
ax.set_title('(b)', x=x, y=y)

ax = plt.subplot(grid[1, 2:4])
ax.plot([0, 0.05, 0.05, 0.15, 0.15, 0.25, 0.25, 0.35,
         0.35] +
        [0.65, 0.65, 0.75, 0.75, 0.85, 0.85, 0.95,
         0.95, 1.00],
        [0, 0.00, 0.05, 0.05, 0.40, 0.40, 0.05, 0.05,
         0.00] +
        [0.00, 0.05, 0.05, 0.40, 0.40, 0.05, 0.05,
         0.00, 0.00],
        label=r'$M_{gas}$-weighted')
bins2 = \
    [0.15, 0.25, 0.75, 0.85]
csp = \
    [0.05, 0.45, 0.55, 0.95]
DGR16, DGR84 = np.interp([0.16, 0.84], csp, bins2)
ax.plot([DGR16] * 2, ylim, label='16%')
ax.plot([DGR84] * 2, ylim, label='84%')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 0.4/0.8])
ax.set_xlabel('DGR')
ax.set_ylabel('P(DGR)')
ax.set_yticks([])
ax.legend()
ax.set_title('(c)', x=x, y=y)
