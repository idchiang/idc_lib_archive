import matplotlib.pyplot as plt
import numpy as np


class PointMover:
    def __init__(self, left_map, right_map):
        self.left_map = left_map
        self.right_map = right_map
        self.fig, self.ax = plt.subplots(1, 2, figsize=(8, 3))
        self.ax[0].set_title('click somewhere on the map')
        #
        # Some map. Replace it with your own.
        #
        cax = self.ax[0].imshow(self.left_map, origin='lower',
                                cmap='inferno')
        plt.colorbar(cax, ax=self.ax[0])
        #
        # Create the clicking point
        #
        self.point, = self.ax[0].plot([0], [0], 'c*')
        plt.show()
        #
        # Link the class to clicking event
        #
        self.cid = \
            self.point.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        #
        # Grab clicking event information
        #
        print('click', event)
        if event.inaxes != self.point.axes:
            return
        j, i = int(round(event.xdata)), int(round(event.ydata))
        #
        # Update pointer position in the first axis
        #
        self.point.set_data(j, i)
        self.point.figure.canvas.draw()
        #
        # Clear the other axes. Avoid nan / inf problem here if you want.
        #
        self.ax[1].clear()
        if not np.isfinite(self.left_map[i, j]):
            return
        #
        # Update the other axes. Replace it with what you want.
        #
        self.ax[1].set_title('y=sin(' + str(i) + 'x+' + str(j) + ')')
        self.ax[1].plot(my_x, np.sin(i * my_x + j))
        plt.draw()


def manual_bkg_removal():
    # 1) Get the clicking template
    # 2) Load current mask. Create one if not existing
    # 3) Plot two panels: with and without masking
    # 4) Save the clicked points
    # 5) Generate small circles and update the mask from the list
    # 6) iterate
    pass


def radial_profile():
    # copy from old plot
    pass


def radial_profiles():
    # 0) use dictionary to save things
    # 1-1) Load HI, H2, fH2, total gas, SFR, SFR mask, dust
    # 1-2) Load radius, R25
    # 1-3) Normalize eveything to a selected radial ring
    # 2) Generate radial profiles with the same radial bins
    # 2-1) radial_profile())
    # 3) Plot everything with radius
    # 3-1) Check how SFR trace CO
    pass


def azimuthal_dependence():
    # 0) use dictionary to save things
    # 1-1) Load H2, SFR, SFR mask
    # 1-2) Load radius, R25
    # 1-3) Normalize eveything to a selected radial ring
    # 2) Check how SFR trace CO in that radial ring
    pass


manual_bkg_removal()
