# This script is used to graph the crystal filters and emotion filters
from crystals import Crystal
from emotions import Emotion, DSM, Protection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib import style

t = np.linspace(-4 * np.pi, 4 * np.pi, 1024)
y = np.sin(2 * t) / 2
base_freq = np.pi

plt.clf()
plt.plot(t, y, c='red', label='Source Wave')
plt.plot(t, Crystal.rose_quartz(t), c='pink', label='Rose Quartz')
plt.plot(t, Crystal.citrine(t), c='yellow', label='Citrine') 
plt.plot(t, Crystal.amethyst(t), c='purple', label='Amethyst')
plt.plot(t, Emotion.joy_filter(y, t), c='aqua', label='Joy')
plt.plot(t, Emotion.sadness_filter(y, t), c='blue', label='Sadness') 
# plt.plot(t, Emotion.fear_filter(y, t), c='maroon', label='Fear')
# plt.plot(t, DSM.anxiety_filter(t), c='black', label='Anxiety')


plt.legend()
plt.show()