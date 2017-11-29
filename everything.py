# from generating import generator
from fitting import fitting
from plotting import plot_dust

method_abbrs = ['EF', 'FB', 'BEMFB', 'FBWD', 'BEMFBFL']
method_abbrs = ['BEMFBFL']

f = 0
p = 1


if f:
    fitting(samples=['NGC5457'],
            cov_mode=True,
            method_abbrs=method_abbrs,
            del_model=False
            )

if p:
    plot_dust(methods=method_abbrs
              )
