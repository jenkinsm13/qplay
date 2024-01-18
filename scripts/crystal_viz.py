from crystals import Crystal, CrystalGrid
import matplotlib.pyplot as plt
import numpy as np
from emotions import Emotion
from scipy.io.wavfile import write

# Time range for ∞§∞
t = np.linspace(-np.pi, np.pi, 4096)
y = np.sin(2*t) / 2
emotion_filter = Emotion.panic(t, y)  # Used for the dynamics of the music

# Creating instances of different crystals
agate = Crystal("Agate", Crystal.agate, properties={"color": "#F5F5DC"}, energy_level=1)
amazonite = Crystal("Amazonite", Crystal.amazonite, properties={"color": "#40E0D0"}, energy_level=1)
amethyst = Crystal("Amethyst", Crystal.amethyst, properties={"color": "#800080"}, energy_level=1)
tigers_eye = Crystal("Tiger's Eye", Crystal.tigers_eye, properties={"color": "#DAA520"}, energy_level=1)
citrine = Crystal("Citrine", Crystal.citrine, properties={"color": "#FFFF00"}, energy_level=1)
bloodstone = Crystal("Bloodstone", Crystal.bloodstone, properties={"color": "#006400"}, energy_level=1)
carnelian = Crystal("Carnelian", Crystal.carnelian, properties={"color": "#A52A2A"}, energy_level=1)

# List of crystals
crystals = [amethyst, tigers_eye]

# Create a CrystalGrid instance
radius = np.pi/2  # Define the radius of the grid
crystal_grid = CrystalGrid(size=len(crystals), radius=radius, crystals=crystals)


# Apply the emotion filter to the time array
modulated_t = t + emotion_filter

# Get the combined waveform with the modulated time array
combined_waveform = crystal_grid.get_waveform(modulated_t, method='additive')

# Normalize the waveform to fit within the range of -1 to 1
normalized_waveform = combined_waveform / np.max(np.abs(combined_waveform))

# Convert the modulated time array to represent 5 seconds
modulated_t = modulated_t * 5 * 192000 / 3 / (2 * np.pi)


# Define the sample rate
sample_rate = 192000  # Hz

# Calculate the number of samples needed for 5 seconds
num_samples = 5 * sample_rate 

# Calculate the number of repeats needed
num_repeats = int(np.ceil(num_samples / len(normalized_waveform)))

# Repeat the waveform data
repeated_waveform = np.tile(normalized_waveform, num_repeats)


# Write the waveform data to a .wav file
write("modulated_agate_amethyst_bloodstone.wav", sample_rate, repeated_waveform)


# Plotting the combined waveform
plt.figure(figsize=(10, 6))
plt.plot(t, combined_waveform, label="Combined Waveform")
plt.title("Combined Waveform from Crystal Grid")
plt.xlabel("Time")
plt.ylabel("Wave Amplitude")
plt.legend()
plt.grid(True)
plt.show()

