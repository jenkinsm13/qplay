# /blocks/qplay_crystals.py

from blocks import HyperBlock, Agent
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import *
import scipy.signal as signal

class Crystal(HyperBlock):
    def __init__(self, name, waveform, properties, position=None, energy_level=1):
        super().__init__(name, thought_vector=None, position=None, energy_level=1)
        self.name = name
        self.waveform = waveform
        self.properties = properties if properties else {}
        self.energy_level = energy_level
        self.position = position

    def filter_wave(self, source_wave):
        filtered_wave = source_wave * self.waveform(source_wave)
        return filtered_wave

    def activate(self, intensity):
        # Crystal activation with adjustable intensity
        # Update internal state or trigger downstream actions
        pass

    def connect_to(self, other_crystal):
        # Establish a connection between two Crystals
        # Enable potential interactions and information exchange
        pass

    def get_waveform(self, t, energy_level):
        return self.waveform(t, energy_level)

    @staticmethod
    # Agate - balance, stability
    def agate(t, energy_level):
        return energy_level * ((1 + 0.1*np.cos(8*t)) * np.exp(-0.01*t))  

    @staticmethod
    # Amazonite - harmony, balance  
    def amazonite(t, energy_level):
        return energy_level * (np.sin(3*t) + 0.5*np.cos(4*t))  

    @staticmethod
    # Amber - vitality, energy
    def amber(t, energy_level): 
        return energy_level * (np.sin(t) * (1.5 + 0.25*np.abs(np.sin(7*t))))

    @staticmethod
    # Amethyst - spiritual awareness  
    def amethyst(t, energy_level):
        return energy_level * (np.sinc(t) * (1 + np.cos(7*t)))

    @staticmethod
    # Apatite - manifestation, creativity  
    def apatite(t, energy_level):
        return energy_level * (np.sin(2*t) * (1.5 + 0.4*np.sin(5*t)))

    @staticmethod
    # Aventurine - positive change
    def aventurine(t, energy_level):
        return energy_level * (np.sin(t) * (1 + 0.1*np.random.rand(t.size)))

    @staticmethod
    # Azurite - insight, intuition
    def azurite(t, energy_level): 
        return energy_level * (1.5* np.sin(2*t) * (0.5 + 0.25*np.random.rand(t.size)))

    @staticmethod
    # Bloodstone - courage, strength  
    def bloodstone(t, energy_level):
        return energy_level * (2*np.sin(t) + np.minimum(np.abs(np.sin(4*t)), 1))

    @staticmethod
    # Calcite - emotional balance
    def calcite(t, energy_level):
        ts = np.mod(t, 2*np.pi)/np.pi  
        return energy_level * (np.sin(2*ts) * np.clip(ts, 0.45, 0.55))

    @staticmethod
    # Carnelian - vitality, sexuality
    def carnelian(t, energy_level):
        return energy_level * (np.sin(t) * (np.abs(np.sin(0.5*t)) + 2))

    @staticmethod
    # Celestite - higher perspective
    def celestite(t, energy_level):
        return energy_level * ((1.5 + np.sin(7*t)) * np.sinc(t/2)) 

    @staticmethod
    # Chrysocolla - communication, expression  
    def chrysocolla(t, energy_level):
        return energy_level * ((np.abs(np.sin(2*t)) + 1)*np.clip(np.sin(5*t), -1, 1))

    @staticmethod
    # Chrysoprase - joy, grace   
    def chrysoprase(t, energy_level):
        return energy_level * (np.clip(np.tan(3*t), -1, 1))

    @staticmethod
    # Citrine - optimism, warmth
    def citrine(t, energy_level):
        return energy_level * (np.sin(2*t) + 0.3*np.cos(5*t))   

    @staticmethod
    # Clear Quartz - amplification, clarity    
    def clear_quartz(t, energy_level):
        base = np.sin(t) 
        return energy_level * (1.2 * base * np.clip(base, -1, 1)**2)   

    @staticmethod
    # Dalmatian Jasper - loyalty, playfulness  
    def dalmatian_jasper(t, energy_level): 
        return energy_level * ((1 + 0.75*np.sin(t)) * (0.75 + 0.25 * np.sin(6*t)))

    @staticmethod
    # Emerald - love, compassion 
    def emerald(t, energy_level):
        return energy_level * (1.75 * np.sin(2*t) * np.exp(-0.05*t**2))   

    @staticmethod
    # Fluorite - stability, harmony
    def fluorite(t, energy_level):
        return energy_level * (np.clip(np.sin(3*t), -0.75, 0.75) + np.cos(t))  

    @staticmethod
    # Garnet - passion, devotion  
    def garnet(t, energy_level):
        return energy_level * (np.sqrt(np.abs(np.sin(2*t))) * (1 + 0.5*np.sin(4*t)))

    @staticmethod
    # Hematite - vitality    
    def hematite(t, energy_level):
        ts = np.clip(np.mod(t, 2*np.pi)/np.pi, 0.1, 0.9) 
        return energy_level * ((ts +  np.sin(t)) * np.sign(np.cos(t)))

    @staticmethod
    # Howlite - calm, relief  
    def howlite(t, energy_level):
        return energy_level * (np.maximum(np.sin(0.5*t), np.cos(t/2)))

    @staticmethod
    # Jade - serenity, wisdom
    def jade(t, energy_level):  
        return energy_level * (np.clip(np.sin(3*t) + 1.1, 0, 1.1) * np.exp(-0.01*t))

    @staticmethod
    # Jasper - grounding, stability   
    def jasper(self, t, energy_level):
        return energy_level * (np.sin(2*t) * 0.75 * np.exp(-0.015*t**2)) 

    @staticmethod
    # Kyanite - healing, meditation  
    def kyanite(t, energy_level):
        return energy_level * (np.sqrt(1 - np.minimum(t**2, 1)) * np.exp(-t**2))  

    @staticmethod
    # Labradorite - transformation, magic  
    def labradorite(t, energy_level):
      return energy_level * (np.sin(2*t) * (1 + 0.5*np.random.rand(1))) 

    @staticmethod
    # Lapis Lazuli - wisdom, intuition  
    def lapis_lazuli(t, energy_level): 
        return energy_level * (1.5 * np.sin(3*t) * (0.5 + 0.1*np.random.rand(t.size)))

    @staticmethod
    # Lepidolite - calm, stress relief    
    def lepidolite(t, energy_level):
        return energy_level * (1.3 * (np.cos(t) + 0.01 * np.clip(np.random.normal(0,1,t.size), -5, 5))) 

    @staticmethod
    # Malachite - spiritual evolution  
    def malachite(t, energy_level):
        return energy_level * (np.sin(2*t) * (0.75 + 0.25*np.sin(7*t))) 

    @staticmethod
    # Menalite - psychic ability 
    def menalite(t, energy_level):
        spikes = np.clip(np.random.poisson(5, t.size) - 5, -1, 1) 
        return energy_level * (np.sin(t) + 0.2*spikes)  

    @staticmethod
    # Mookaite - stability, strength  
    def mookaite(t, energy_level):
        return energy_level * (1.5*np.sin(3*t) + 0.2*np.minimum(np.abs(np.sin(4*t)), 1))

    @staticmethod
    # Moonstone - fertility, creativity 
    def moonstone(t, energy_level):
        return energy_level * ((1 + 0.2*np.sin(7*t)) * (0.5 + 0.15*np.random.rand(t.size)))

    @staticmethod
    # Obsidian - protection, cleansing    
    def obsidian(t, energy_level):
        sign = np.sign(np.cos(3*t))
        return energy_level * (sign * np.minimum(1, np.abs(np.sin(2*sign*t))))  

    @staticmethod
    # Onyx - focusing, persistence  
    def onyx(t, energy_level): 
        return energy_level * (1.2*np.abs(np.sin(t)) * np.exp(-0.025*t**2))

    @staticmethod
    # Opal - imagination, magic  
    def opal(t, energy_level):
        return energy_level * (np.sin(2*t) * (1 + np.random.rand(t.size))) 

    @staticmethod
    # Peridot - vitality, cleansing
    def peridot(t, energy_level):
        return energy_level * (1.5 * np.abs(np.sin(t)) * (1 + 0.25 * np.sin(4*t)))  

    @staticmethod
    # Quartz - amplification, programming  
    def quartz(t, energy_level):
        base = np.sin(t)
        return energy_level * (1.2 * base * np.clip(base, -1, 1)**2)    

    @staticmethod
    # Rhodochrosite - emotional healing   
    def rhodochrosite(self, t, energy_level): 
        wave = np.sin(3*t)
        healed = savgol_filter(wave, 101, 2)
        return energy_level * (wave - np.clip(np.abs(wave - healed), 0, 0.5))   

    @staticmethod
    # Rhodonite - emotional healing  
    def rhodonite(self, t, energy_level):
        return energy_level * (0.8*self.rhodochrosite(t) + 0.2*self.jasper(t)) 

    @staticmethod
    # Rose Quartz - unconditional love   
    def rose_quartz(t, energy_level):
      return energy_level * (np.sin(t) * (1 + 0.5*np.cos(3*t)) * np.exp(-0.1*t))

    @staticmethod
    # Ruby - passion, focus  
    def ruby(t, energy_level):
        return energy_level * (1.2*np.sqrt(np.abs(np.sin(3*t))) * (0.5 + 0.25* np.sign(np.sin(4*t))))  

    @staticmethod
    # Smokey Quartz - cleansing, grounding   
    def smokey_quartz(t, energy_level):
        return energy_level * (np.sign(np.sin(3*t)) * np.sin(t) * np.exp(-0.01*t))  

    @staticmethod
    # Sodalite - calm, harmony 
    def sodalite(t, energy_level):
        return energy_level * (np.sqrt(np.abs(np.sin(0.5*t))))

    @staticmethod
    # Sunstone - positivity, expression
    def sunstone(t, energy_level):  
        offset = 0.3 * np.sin(4*t)
        return energy_level * (offset + np.clip(np.sin(t + offset), -0.8, 0.8))

    @staticmethod
    # Tiger's Eye - clarity, confidence  
    def tigers_eye(t, energy_level):
        return energy_level * (np.sqrt(np.abs(np.sin(t))) * (1.0 + 0.1*np.random.rand(t.size)))   

    @staticmethod
    # Topaz - manifestation, creativity   
    def topaz(t, energy_level):  
      return energy_level * (1.5*np.sqrt(np.abs(np.sin(2*t))) * (1 + 0.5*np.sin(6*t)))

    @staticmethod
    # Tourmaline - balance, vibration  
    def tourmaline(t, energy_level):
        return energy_level * (np.sin(2*t) * np.exp(-0.01*t) + np.sin(4*t)) 

    @staticmethod
    # Turquoise - wisdom, luck    
    def turquoise(t, energy_level):
        r = 0.25 * np.random.rand(t.size)
        offset = 0.75 + r
        return energy_level * (offset + np.sin(3 * (t + r)))  

    @staticmethod
    # Unakite - visualization, psychic vision  
    def unakite(t, energy_level):
        r = 5*np.random.rand(1)   
        wave = np.sqrt(np.abs(np.sin(t + r)))  
        env = np.minimum(t/np.pi, np.ones_like(t))
        return energy_level * (wave * env * (1 + np.sin(6*t)))


class CrystalChannel(Crystal):
    "Handles direct crystal invocation"

    def __init__(self, crystal):
        super().__init__(f"Channeling {crystal}") 
        self.crystal = crystal
        
    def inkove_crystal(self, crystal, t):
        """Invoke held crystal on waveform t"""
        filtered_t = crystal[self.crystal](t)
        return filtered_t
    
    def get_waveform(self, t):
        return self.crystal.get_waveform(t)

class CrystalGrid(Agent):
    "Constructs crystal arrangements"  
  
    def __init__(self, size, radius, crystals=None, name="Crystal Grid"):
        super().__init__(name, thought_vector=None, state="entangled")
        self.size = size
        self.radius = radius
        self.crystals = crystals if crystals else []
        self.assign_positions()

    def _map_grid(self):
        """Assign crystal positions"""
        angles = np.linspace(0, 2*np.pi, len(self.crystals))
        positions = [[self.radius*np.cos(a), 0, self.radius*np.sin(a)] for a in angles]  
        # Create a mapping of crystals to positions
        crystal_to_position = dict(zip(self.crystals, positions))
        # Create a mapping of slots to crystals
        slot_to_crystal = {i: crystal for i, crystal in enumerate(self.crystals)}
        return crystal_to_position, slot_to_crystal

    # Circular Grid and Wave Positioning
    def place_crystals_on_grid(self, num_crystals, t, emotion_filter):
        """Place crystals in an evenly spaced circular grid."""
        angles = np.linspace(0, 2 * np.pi, num_crystals, endpoint=False)
        # Apply the emotion filter to the positions
        positions = [(np.cos(angle), emotion_filter(np.sin(angle), t)) for angle in angles]
        return positions

    def distance_from_center(self, position):
        """Calculate the distance of a point from the center."""
        x, y = position
        return np.sqrt(x**2 + y**2)

    # Wave Combination Based on Position
    def combine_crystals_based_on_position(self, waves, positions):
        """Combine waves based on their positions."""
        combined_wave = np.zeros_like(waves[0])
        for wave, position in zip(waves, positions):
            distance = self.distance_from_center(position)
            # Influence of position on the wave can be adjusted here
            wave_influenced = wave * (1 / (1 + distance))
            combined_wave += wave_influenced
        return combined_wave

    # Combining Multiple Waveforms
    def combine_crystals(self, method, crystals, positions, waves):
        
    # Combine crystals based on specified method
        if method == 'additive':
            return self._additive_blend(crystals, positions)
        elif method == 'product':
            return self._multiply_blend(crystals, positions)
        elif method == 'convolution':
            return self._convolution_blend(crystals, positions)
        elif method == 'interference':
            return self._interference_pattern(crystals, positions)
        elif method == 'entangled':
            return self._entangle_waves(crystals, positions)
    
    def _entangle_waves(self, waves, positions):
        """Combine waves based on their positions using entanglement logic."""
        combined_wave = np.zeros_like(waves[0])
        for wave, position in zip(waves, positions):
            # Normalize the position scale to -2π to 2π
            normalized_position = (np.linalg.norm(position) / self.radius) * 4 * np.pi - 2 * np.pi
            combined_wave += wave * normalized_position
        return combined_wave / len(waves)

    def _additive_blend(self, waves, positions):
        """Combine waves additively with position-based weighting."""
        combined_wave = np.zeros_like(waves[0])
        for wave, position in zip(waves, positions):
            normalized_position = (np.linalg.norm(position) / self.radius) * 4 * np.pi - 2 * np.pi
            combined_wave += wave * normalized_position
        return combined_wave / len(waves)

    def _multiply_blend(self, waves, positions):
        """Multiply waves with position-based modification."""
        combined_wave = np.ones_like(waves[0])
        for wave, position in zip(waves, positions):
            normalized_position = (np.linalg.norm(position) / self.radius) * 4 * np.pi - 2 * np.pi
            combined_wave *= wave * normalized_position
        return combined_wave

    def _convolution_blend(self, waves, positions):
        """Convolve waves with position-based influence."""
        combined_wave = np.zeros_like(waves[0])
        for wave, position in zip(waves, positions):
            normalized_position = (np.linalg.norm(position) / self.radius) * 4 * np.pi - 2 * np.pi
            combined_wave = signal.convolve(combined_wave, wave * normalized_position, mode="same")
        return combined_wave / len(waves)
        
    def _interference_pattern(self, waves, positions):
        """Create interference patterns based on wave and position."""
        x = np.linspace(-np.pi, np.pi, 1024)
        combined_wave = np.zeros_like(x)
        for wave, position in zip(waves, positions):
            normalized_position = (np.linalg.norm(position) / self.radius) * 4 * np.pi - 2 * np.pi
            combined_wave += np.sin(5 * normalized_position * np.linalg.norm([x, wave]))
        return combined_wave
    
    """
    def _entangle_waves(self, waves, positions):
        return self.combine_crystals_based_on_position(waves, positions)

    def _additive_blend(self, waves):
        return sum(waves) * 1/len(waves)

    def _multiply_blend(self, waves):  
        return np.prod(np.array(waves), axis=0) 

    def _convolution_blend(self, waves):
        conv = signal.convolve(waves[0], waves[1], mode="same")
        return conv
        
    def _interference_pattern(self, waves):
        Superpose like sound/light waves  
        x = np.linspace(-np.pi, np.pi, 1024)
        z = sum([np.sin(5 * np.linalg.norm([x_i, wave])) 
                 for x_i, wave in zip(x, waves)]) 
        return z
    """


    def combine_multiple_waves(self, crystal_functions, t, energy_level, method, num_crystals=9):
        positions = self.place_crystals_on_grid(num_crystals)
        crystals = [crystal_function(t, energy_level) for crystal_function in crystal_functions]
        return self.combine_waves(method, crystals, positions)
    
    def get_waveform(self, t, method):
        """
        Orchestrating method:
            1. Gets individual crystal waveforms  
            2. Combines them as a composite grid waveform based on their positions
            3. Returns final waveform
        """
        # Generate waveforms and positions for each crystal
        waves = []
        positions = []
        for crystal in self.crystals:
            wave = crystal.get_waveform(t, energy_level=crystal.energy_level)
            waves.append(wave)
            positions.append(crystal.position)

        # Combine the waveforms based on positions and the specified method
        if method == 'product':
            combined_waveform = self._multiply_blend(waves, positions)
        elif method == 'additive':
            combined_waveform = self._additive_blend(waves, positions)
        elif method == 'convolution':
            combined_waveform = self._convolution_blend(waves, positions)
        elif method == 'interference':
            combined_waveform = self._interference_pattern(waves, positions)
        elif method == 'entangled':
            combined_waveform = self._entangle_waves(waves, positions)
        else:
            combined_waveform = np.zeros_like(t)  # Default to zero waveform if method is unknown

        return combined_waveform
    
    def assign_positions(self):
        """Assign positions to crystals in a circular grid on the xz plane."""
        angles = np.linspace(0, 2 * np.pi, self.size, endpoint=False)
        for angle, crystal in zip(angles, self.crystals):
            x = self.radius * np.cos(angle)
            z = self.radius * np.sin(angle)
            crystal.position = (x, 0, z)  # Position in xz plane


    def add_crystal(self, crystal):
        self.crystals.append(crystal)
        # Assign a position to the new crystal
        self.assign_positions()

    def update_radius(self, new_radius):
        self.radius = new_radius
        self.assign_positions()

    def visualize(self, as_shapes=False, label_type=None):
            """Visualize the grid of crystals."""
            # Extract x and z coordinates from positions
            x_coords = [crystal.position[0] for crystal in self.crystals]
            z_coords = [crystal.position[2] for crystal in self.crystals]

            # Extract colors of the crystals
            colors = [crystal.properties.get("color", "black") for crystal in self.crystals]

            # Create a scatter plot
            plt.figure(figsize=(8, 8))
            plt.scatter(x_coords, z_coords, c=colors, s=100)  # Adjust size as needed

            # Add labels
            if label_type == 'name':
                for crystal, x, z in zip(self.crystals, x_coords, z_coords):
                    plt.text(x, z, crystal.name, horizontalalignment='center')

            # Set plot limits and labels
            plt.xlim(-self.radius-1, self.radius+1)
            plt.ylim(-self.radius-1, self.radius+1)
            plt.xlabel("X-axis")
            plt.ylabel("Z-axis")
            plt.title("Crystal Grid Visualization")
            plt.grid(True)
            plt.axhline(0, color='black',linewidth=0.5)
            plt.axvline(0, color='black',linewidth=0.5)
            plt.show()