from matplotlib import pyplot as plt  
from matplotlib import animation  
import matplotlib.image as mgimg
import numpy as np
import os


fig = plt.figure()
ax = plt.gca()
ax = plt.gca()
path = "/Users/rubencr/Desktop/ML_notebooks/ML_models/ML_tutorials_github/generated_images"
images = [files for files in os.listdir(path) if files.endswith("png")]
first = plt.imread(os.path.join(path, images[0]))
im = ax.imshow(first, animated=True)


def animate(i):
    imc = plt.imread(os.path.join(path, images[i]))
    im.set_data(imc)
    return im,

ani = animation.FuncAnimation(fig, animate, frames=len(images), repeat=True, interval=500, repeat_delay=1000)
ani.save('animation.gif', writer='imagemagick')
plt.show()
