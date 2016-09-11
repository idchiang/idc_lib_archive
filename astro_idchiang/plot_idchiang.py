import matplotlib.pyplot as plt

def imshowid(image):
    plt.imshow(image, origin = 'lower')
    plt.colorbar()