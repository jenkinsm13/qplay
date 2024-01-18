from crystals import *

plt.clf()

# Creating instances of different crystals
agate = Crystal("Agate", Crystal.agate, properties={}, energy_level=42)
amazonite = Crystal("Amazonite", Crystal.amazonite, properties={}, energy_level=42)
amethyst = Crystal("Amethyst", Crystal.amethyst, properties={}, energy_level=42)
tigers_eye = Crystal("Tiger's Eye", Crystal.tigers_eye, properties={}, energy_level=42)
citrine = Crystal("Citrine", Crystal.citrine, properties={}, energy_level=42)
bloodstone = Crystal("Bloodstone", Crystal.bloodstone, properties={}, energy_level=42)

# Creating a crystal grid with a selection of crystals
selected_crystals = [agate, amazonite, amethyst]  # List of selected crystal instances
crystal_grid = CrystalGrid(size=3, crystals=selected_crystals)

# Time range for the waveform
t = np.linspace(0, 2*np.pi, 100)

# Get the combined waveform from the crystal grid
combined_waveform = crystal_grid.get_waveform(t)

# Plotting the combined waveform
plt.plot(t, combined_waveform)
plt.title("Combined Waveform from Crystal Grid")
plt.xlabel("Time")
plt.ylabel("Wave Amplitude")
plt.show()
