# /blocks/qplay_crystals.py

import qplay_blocks
from qplay_blocks import HyperBlock
import numpy as np
import matplotlib.pyplot as plt

class Crystal(HyperBlock):
    def __init__(self, name, wave_function, properties):
        super().__init__(name, thought_vector=None, state="superposition")
        self.wave_function = wave_function
        self.properties = properties

    def filter_wave(self, source_wave):
        filtered_wave = source_wave * self.wave_function(source_wave)
        return filtered_wave

    def activate(self, intensity):
        # Crystal activation with adjustable intensity
        # Update internal state or trigger downstream actions
        pass

    def connect_to(self, other_crystal):
        # Establish a connection between two Crystals
        # Enable potential interactions and information exchange
        pass

    # Agate - balance, stability
    def agate(t):
        return (1 + 0.1*np.cos(8*t)) * np.exp(-0.01*t)  

    # Amazonite - harmony, balance  
    def amazonite(t):
        return np.sin(3*t) + 0.5*np.cos(4*t)   

    # Amber - vitality, energy
    def amber(t): 
        return np.sin(t) * (1.5 + 0.25*np.abs(np.sin(7*t)))

    # Amethyst - spiritual awareness  
    def amethyst(t):
        return np.sinc(t) * (1 + np.cos(7*t))  

    # Apatite - manifestation, creativity  
    def apatite(t):
        return np.sin(2*t) * (1.5 + 0.4*np.sin(5*t))

    # Aventurine - positive change
    def aventurine(t):
        return np.sin(t) * (1 + 0.1*np.random.rand(t.size))

    # Azurite - insight, intuition
    def azurite(t): 
        return 1.5* np.sin(2*t) * (0.5 + 0.25*np.random.rand(t.size))

    # Bloodstone - courage, strength  
    def bloodstone(t):
        return 2*np.sin(t) + np.minimum(np.abs(np.sin(4*t)), 1)

    # Calcite - emotional balance
    def calcite(t):
        ts = np.mod(t, 2*np.pi)/np.pi  
        return np.sin(2*ts) * np.clip(ts, 0.45, 0.55)

    # Carnelian - vitality, sexuality
    def carnelian(t):
        return np.sin(t) * (np.abs(np.sin(0.5*t)) + 2)

    # Celestite - higher perspective
    def celestite(t):
        return (1.5 + np.sin(7*t)) * np.sinc(t/2) 

    # Chrysocolla - communication, expression  
    def chrysocolla(t):
        return (np.abs(np.sin(2*t)) + 1)*np.clip(np.sin(5*t), -1, 1)

    # Chrysoprase - joy, grace   
    def chrysoprase(t):
        return np.clip(np.tan(3*t), -1, 1)

    # Citrine - optimism, warmth
    def citrine(t):
        return np.sin(2*t) + 0.3*np.cos(5*t)   

    # Clear Quartz - amplification, clarity    
    def clear_quartz(t):
        base = np.sin(t) 
        return 1.2 * base * np.clip(base, -1, 1)**2   

    # Dalmatian Jasper - loyalty, playfulness  
    def dalmatian_jasper(t): 
        return (1 + 0.75*np.sin(t)) * (0.75 + 0.25 * np.sin(6*t))

    # Emerald - love, compassion 
    def emerald(t):
        return 1.75 * np.sin(2*t) * np.exp(-0.05*t**2)   

    # Fluorite - stability, harmony
    def fluorite(t):
        return np.clip(np.sin(3*t), -0.75, 0.75) + np.cos(t)  

    # Garnet - passion, devotion  
    def garnet(t):
        return np.sqrt(np.abs(np.sin(2*t))) * (1 + 0.5*np.sin(4*t))

    # Hematite - vitality    
    def hematite(t):
        ts = np.clip(np.mod(t, 2*pi)/pi, 0.1, 0.9) 
        return (ts +  np.sin(t)) * np.sign(np.cos(t))

    # Howlite - calm, relief  
    def howlite(t):
        return np.maximum(np.sin(0.5*t), np.cos(t/2))

    # Jade - serenity, wisdom
    def jade(t):  
        return np.clip(np.sin(3*t) + 1.1, 0, 1.1) * np.exp(-0.01*t)

    # Jasper - grounding, stability   
    def jasper(t):
        return np.sin(2*t) * 0.75 * np.exp(-0.015*t**2) 

    # Kyanite - healing, meditation  
    def kyanite(t):
        return np.sqrt(1 - np.minimum(t**2, 1)) * np.exp(-t**2)  

    # Labradorite - transformation, magic  
    def labradorite(t):
      return np.sin(2*t) * (1 + 0.5*np.random.rand(1)) 

    # Lapis Lazuli - wisdom, intuition  
    def lapis_lazuli(t): 
        return 1.5 * np.sin(3*t) * (0.5 + 0.1*np.random.rand(t.size))

    # Lepidolite - calm, stress relief    
    def lepidolite(t):
        return 1.3 * (np.cos(t) + 0.01 * np.clip(np.random.normal(0,1,t.size), -5, 5) ) 

    # Malachite - spiritual evolution  
    def malachite(t):
        return np.sin(2*t) * (0.75 + 0.25*np.sin(7*t)) 

    # Menalite - psychic ability 
    def menalite(t):
        spikes = np.clip(np.random.poisson(5, t.size) - 5, -1, 1) 
        return np.sin(t) + 0.2*spikes  

    # Mookaite - stability, strength  
    def mookaite(t):
        return 1.5*np.sin(3*t) + 0.2*np.minimum(np.abs(np.sin(4*t)), 1)

    # Moonstone - fertility, creativity 
    def moonstone(t):
        return (1 + 0.2*np.sin(7*t)) * (0.5 + 0.15*np.random.rand(t.size))

    # Obsidian - protection, cleansing    
    def obsidian(t):
        sign = np.sign(np.cos(3*t))
        return sign * np.minimum(1, np.abs(np.sin(2*sign*t)))  

    # Onyx - focusing, persistence  
    def onyx(t): 
        return 1.2*np.abs(np.sin(t)) * np.exp(-0.025*t**2)

    # Opal - imagination, magic  
    def opal(t):
        return np.sin(2*t) * (1 + np.random.rand(t.size)) 

    # Peridot - vitality, cleansing
    def peridot(t):
        return 1.5 * np.abs(np.sin(t)) * (1 + 0.25 * np.sin(4*t))  

    # Quartz - amplification, programming  
    def quartz(t):
        base = np.sin(t)
        return 1.2 * base * np.clip(base, -1, 1)**2    

    # Rhodochrosite - emotional healing   
    def rhodochrosite(t): 
        wave = np.sin(3*t)
        healed = savgol_filter(wave, 101, 2)
        return wave - np.clip(np.abs(wave - healed), 0, 0.5)   

    # Rhodonite - emotional healing  
    def rhodonite(t):
        return 0.8*rhodochrosite(t) + 0.2*jasper(t) 

    # Rose Quartz - unconditional love   
    def rose_quartz(t):
      return np.sin(t) * (1 + 0.5*np.cos(3*t)) * np.exp(-0.1*t)

    # Ruby - passion, focus  
    def ruby(t):
        return 1.2*np.sqrt(np.abs(np.sin(3*t))) * (0.5 + 0.25* np.sign(np.sin(4*t)) )  

    # Smokey Quartz - cleansing, grounding   
    def smokey_quartz(t):
        return np.sign(np.sin(3*t)) * np.sin(t) * np.exp(-0.01*t)  

    # Sodalite - calm, harmony 
    def sodalite(t):
        return np.sqrt(np.abs(np.sin(0.5*t)))

    # Sunstone - positivity, expression
    def sunstone(t):  
        offset = 0.3 * np.sin(4*t)
        return offset + np.clip(np.sin(t + offset), -0.8, 0.8)

    # Tiger's Eye - clarity, confidence  
    def tigers_eye(t):
        return np.sqrt(np.abs(np.sin(t))) * (1.0 + 0.1*np.random.rand(t.size))   

    # Topaz - manifestation, creativity   
    def topaz(t):  
      return 1.5*np.sqrt(np.abs(np.sin(2*t))) * (1 + 0.5*np.sin(6*t))

    # Tourmaline - balance, vibration  
    def tourmaline(t):
        return np.sin(2*t) * np.exp(-0.01*t) + np.sin(4*t) 

    # Turquoise - wisdom, luck    
    def turquoise(t):
        r = 0.25 * np.random.rand(t.size)
        offset = 0.75 + r
        return offset + np.sin(3 * (t + r))  

    # Unakite - visualization, psychic vision  
    def unakite(t):
        r = 5*np.random.rand(1)   
        wave = np.sqrt(np.abs(np.sin(t + r)))  
        env = np.minimum(t/np.pi, np.ones_like(t))
        return wave * env * (1 + np.sin(6*t))


class CrystalChannel(Crystal):
    "Handles direct crystal invocation"

    def __init__(self, crystal):
        super().__init__(f"Channeling {crystal}") 
        self.crystal = crystal
        
    def inkove_crystal(self, crystal, t):
        """Invoke held crystal on waveform t"""
        filtered_t = crystal_waves[self.crystal](t)
        return filtered_t

class CrystalGrid(Crystal):
    "Constructs crystal arrangements"  
  
    def __init__(self, size, crystals):
        super().__init__(f"Grid {size}") 
        self.size = size
        self.crystals = crystals
        self.positions = self._map_to_grid()

    def _entangle_waves(self, waves, positions): 
        """Simple additive blending as example"""
        return sum(waves)

    def _map_grid(self):
        """Assign crystal positions"""
        angles = np.linspace(0, 2*pi, self.size + 1)[:-1]
        positions = [[cos(a), 0, sin(a)] for a in angles]  
        return dict(zip(self.crystals, positions))

    # Circular Grid and Wave Positioning

    def place_crystals_on_grid(num_crystals):
        """Place crystals in an evenly spaced circular grid."""
        angles = np.linspace(0, 2 * np.pi, num_crystals, endpoint=False)
        positions = [(np.cos(angle), np.sin(angle)) for angle in angles]
        return positions

    def distance_from_center(position):
        """Calculate the distance of a point from the center."""
        x, y = position
        return np.sqrt(x**2 + y**2)

    # Wave Combination Based on Position

    def combine_waves_based_on_position(waves, positions):
        """Combine waves based on their positions."""
        combined_wave = np.zeros_like(waves[0])
        for wave, position in zip(waves, positions):
            distance = distance_from_center(position)
            # Influence of position on the wave can be adjusted here
            wave_influenced = wave * (1 / (1 + distance))
            combined_wave += wave_influenced
        return combined_wave

    # Combining Multiple Waveforms

    def combine_multiple_waves(wave_functions, t, num_crystals=9):
        positions = place_crystals_on_grid(num_crystals)
        waves = [wave_function(t) for wave_function in wave_functions]
        return combine_waves_based_on_position(waves, positions)

    def get_waveform(self, t):
      """
      Orchestrating method:
         1. Gets individual crystal waves  
         2. Entangles them as a composite grid waveform  
         3. Returns final waveform
      """
      crystal_waves = []
      for crystal in self.crystals:
          wave = crystal_waves[crystal](t)
          crystal_waves.append(wave)
      entangled = self._entangle_waves(waves, positions)  
 
      return entangled
