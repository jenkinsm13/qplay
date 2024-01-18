# /blocks/qplay_emotions.py

from core import *
from blocks import Block
import numpy as np
from scipy.signal import savgol_filter
import scipy.signal as signal

# Function to create a smooth, sinusoidal filter for Joy
# Function to create a low-frequency, elongated waveform for Sadness
# Function to create a high-frequency, erratic filter for Fear
# Function to create a sporadic spikes filter for Surprise
# Function to create an uneven, asymmetric waveform for Disgust
# Function to create a consistently patterned, smooth waveform for Trust
# Function to create a gradually building waveform for Anticipation
# Function to create a soft, flowing waveform for Love
# Function to create a monotonous, flat waveform for Boredom
# Function to create a waveform with variable frequencies for Curiosity
# Implement a Savitzky-Golay (SG) smoothing filter for lower frequency noise.
# Implement a Butterworth filter for higher frequency noise.

class Emotion(Block):

    def __init__(self, emotion_filter):
         super().__init__(emotion_filter)
         self.emotion_filter = emotion_filter

# Positivity/Joy

    # We can add methods here that are specific to groups of emotions.
    # For example, a method that handles positive-specific reactions to events.
    def perceive_positive_event(self, event):
        # Implementation for perceiving positive events
        pass

    @staticmethod
    def joy(t, y):
        return np.sin(t * y)
    
    @staticmethod
    def bliss(t, y):
        return y + np.random.rand(len(y))
    
    @staticmethod
    def delight(t, y):
        return np.sin(y) + 1
    
    @staticmethod
    def cheer(t, y):
        return np.sqrt(y) + 0.5*np.sin(3*t)
    
    @staticmethod
    def comfort(t, y):
        return savgol_filter(y, window_length=51, polyorder=3)
    
    @staticmethod
    def valuable(t, y):
        return y * (1 + 0.5 * np.sin(2 * t)) * (1 + 0.3 * np.sin(4 * t))

# Negativity/Sadness

    def perceive_negative_event(self, event):
        # Implementation for perceiving negative events
        pass

    @staticmethod
    def sadness(t, y):
        return savgol_filter(y, window_length=101, polyorder=3)
    
    @staticmethod
    def grief(t, y):
        return y - abs(np.random.rand(len(y)))
    
    @staticmethod
    def sorrow(t, y):
        return savgol_filter(1 - y, window_length=51, polyorder=3)
    
    @staticmethod
    def melancholy(t, y):
        return y * (1 - 0.5*np.sin(3*t))
    
    @staticmethod
    def mourning(y, t):
        smoothed = savgol_filter(np.float64(y), window='symmetric', polyorder=2, values=35, delta=2)
        return np.where(y >= 0.7*smoothed, y-0.15*smoothed, y+0.15*smoothed)  # Generate sporadic fluctuations from smooth data points.
    
    @staticmethod
    def improved_mourning(t, y):
        # Smooth the input signal
        smoothed = savgol_filter(y, window_length=100, polyorder=2)
        # Generate sporadic fluctuations
        randomness = np.random.randn(len(y)) * 0.1
        # Define threshold as asymmetric waveform
        threshold = 0.7 + 0.1 * np.sin(t)
        # Create dips relative to smooth baseline
        dipped = np.where(y >= threshold * smoothed,
                          y - 0.15 * smoothed + randomness,
                          y + 0.05 * smoothed + randomness)
        return dipped
    
    @staticmethod
    def longing(t, y):
        return y * (1 + 0.3 * np.sin(0.8 * t)) * np.exp(-0.1 * t)
    
    @staticmethod
    def amusement(t, y):
        return y * (1 + 0.5 * np.sin(4 * t)) * (1 + 0.2 * np.random.randn(len(y)))

# Energy/Vitality

    def perceive_vitality_event(self, event):
        # Implementation for perceiving vitality-related events
        pass

    @staticmethod
    def vigor(t, y):
        return y * (np.sin(t) + 1)
    
    @staticmethod
    def vivacity(t, y):
        return y * np.sin(t) * (0.5 + 0.5*np.sin(5*t))
    
    @staticmethod
    def enthusiasm(t, y):
        return y * (1.5 + 0.5*np.abs(np.sin(3*t)))
    
    @staticmethod
    def focus(t, y):
        return y * np.exp(-0.1*t)
    
    @staticmethod
    def awareness(t, y):
        return np.abs(np.fft.ifft(np.fft.fft(y)**2))
    
    @staticmethod
    def excitement(t, y):
        return y * (np.sin(3*t) + 2)
    
    @staticmethod
    def liveliness(t, y):
        return 2*np.sin(4*t)*(y+0.5)
    
    @staticmethod
    def playfulness(t, y):
        return (y+1) * np.sin(5*t)
    
    @staticmethod
    def wistfulness(t, y):
        return y * (1 + 0.2 * np.sin(base_freq * y)) * np.exp(-0.1 * t)

# Calmness/Peace

    def perceive_peaceful_event(self, event):
        # Implementation for perceiving peaceful events
        pass

    @staticmethod
    def calm(t, y):
        return savgol_filter(y, 101, 3)
    
    @staticmethod
    def peace(t, y):
        return np.minimum(y, savgol_filter(np.abs(y), 31, 2))
    
    @staticmethod
    def zen(t, y):
        return np.minimum(y, savgol_filter(y, 101, 2))
    
    @staticmethod
    def tranquility(t, y):
        return savgol_filter(np.maximum(y,0), 201, 3)
    
    @staticmethod
    def harmony(t, y):
        return y * (1 + 0.75*np.cos(3*t))
    
    @staticmethod
    def boredom(t, y):
        return np.full(y.shape, np.mean(y))
    
    @staticmethod
    def contentment(t, y):
        return 0.8 * y + 0.2 * np.sin(0.5 * base_freq * y)
    
    @staticmethod
    def relaxation(t, y):
        return np.minimum(y, savgol_filter(np.abs(y), 51, 3))
    
    @staticmethod
    def serenity(t, y):
        return np.sin(0.8 * base_freq * y) * np.exp(-0.05 * t)

# Interest

    def perceive_interest_event(self, event):
        # Implementation for perceiving high-interest events
        pass
    
    @staticmethod
    def creativity(t, y):
        return y * (1 + 0.5*np.sin(3*t))
    
    @staticmethod
    def anticipation(t, y):
        return y ** 2
    
    @staticmethod
    def fascination(t, y):
        return y * np.cos(5*t)
    
    @staticmethod
    def wonder(t, y):
        return y + 0.1*np.random.normal(0, 1, len(y))
    
    @staticmethod
    def reverence(t, y):
        return savgol_filter(y*np.sin(3*t), 101, 3)
    
    @staticmethod
    def awe(t, y):
        return np.sin(base_freq * y) * (1 + 0.4 * np.sin(0.4 * t))
    
    @staticmethod
    def curiousity(t, y):
        return savgol_filter(y, window_length=51, polyorder=2) * np.cos(y)
    
    @staticmethod
    def nostalgia(t, y):
        return savgol_filter(y, window_length=71, polyorder=3) * 0.8
    
    @staticmethod
    def inspiration(t, y):
        return y + np.random.randn(len(y)) * 0.1 * np.maximum(0, y)
    
    @staticmethod
    def admiration(t, y):
        return 0.8 * y + 0.2 * np.sin(2 * base_freq * y)

# Confidence

    def perceive_confidence_event(self, event):
        # Implementation for perceiving confidence-related events
        pass

    @staticmethod
    def confidence(t, y):
        return y + 0.1 * np.random.randn(len(y))
    
    @staticmethod
    def pride(t, y):
        return y + np.tanh(y)
    
    @staticmethod
    def boldness(t, y):
        return y + 0.5*np.random.rand(len(y))
    
    @staticmethod
    def grit(t, y):
        return np.abs(y) + 0.2
    
    @staticmethod
    def determination(t, y):
        return y * (1 + np.tanh(t))
    
    @staticmethod
    def motivation(t, y):
        return y * (1.5 + 0.5 * np.sin(2*t))

# Love/Connection

    def perceive_connection_event(self, event):
        # Implementation for perceiving connection events
        pass
    
    @staticmethod
    def brotherly(t, y):
        return np.maximum(y, np.sin(3*t))
    
    @staticmethod
    def warmth(t, y):
        return savgol_filter(y, 15, 2) + 0.1
    
    @staticmethod
    def intimacy(t, y):
        return np.sqrt(np.abs(y)) * np.exp(-0.05*t)
    
    @staticmethod
    def love(t, y):
        return np.sqrt(np.abs(y))
    
    @staticmethod
    def intimate(t, y):
        return np.sqrt(np.abs(y)) * np.exp(-0.05*t)
    
    @staticmethod
    def affection(t, y):
        return np.maximum(np.sin(3*t), y) + 0.3
    
    @staticmethod
    def friendly(t, y):
        return savgol_filter(y, 15, 2) + 0.1
    
    @staticmethod
    def passionate(t, y):
        return (y**2) * (1 + np.sin(t))

# Surprise

    def perceive_surprise_event(self, event):
        # Implementation for perceiving surprise events
        pass
    
    @staticmethod
    def epiphany(t, y):
        return y * (1 + np.random.randint(0,5,len(y))*0.1)
    
    @staticmethod
    def astonishment(t, y):
        return np.tanh(np.sin(3*y))
    
    @staticmethod
    def amazement(t, y):
        return y + 0.05*np.random.randint(2,10,len(y))
    
    @staticmethod
    def surprise(t, y):
        return y * (1 + np.random.randint(-1, 2, y.shape) * 0.3)

# Fear

    def perceive_fear_event(self, event):
        # Implementation for perceiving fear events
        pass

    @staticmethod
    def fear(t, y):
        return np.random.normal(y, 0.1)
    
    @staticmethod
    def horror(t, y):
        return y * (1 - 0.5*np.random.rand(len(y)))
    
    @staticmethod
    def terror(t, y):
        return 0.5*y * (1 + 0.2*np.random.randint(2,10,len(y)))
    
    @staticmethod
    def panic(t, y):
        return y + 0.1*np.random.normal(0, 3, len(y))

# Trust

    def perceive_trust_event(self, event):
        # Implementation for perceiving trusting events
        pass

    @staticmethod
    def trust(t, y):
        return np.tanh(y)
    
    @staticmethod
    def faith(t, y):
        return y + np.minimum(y, np.ones(len(y))*1.2)
    
    @staticmethod
    def devotion(t, y):
        return y + 0.5 + 0.25*np.cos(3*t)
    
    @staticmethod
    def loyalty(t, y):
        return y + 0.2*np.cos(6*t)
    
    @staticmethod
    def hope(t, y):
        return savgol_filter(y, window_length=15, polyorder=2) + np.maximum(0, y)
    
    @staticmethod
    def optimism(t, y):
        return np.sqrt(np.abs(y)) + 0.3
    
    @staticmethod
    def gratitude(t, y):
        return savgol_filter(y, window_length=31, polyorder=3) * 1.2
    
    @staticmethod
    def acceptance(t, y):
        return y + 0.1*np.random.rand(len(y))

# Disgust

    def perceive_disgust_event(self, event):
        # Implementation for perceiving disgusting events
        pass

    @staticmethod
    def disgust(t, y):
        return np.where(y > 0, y * 1.5, y * 0.5)
    
    @staticmethod
    def loathing(t, y):
        return y - np.abs(0.2*np.random.rand(len(y)))
    
    @staticmethod
    def revulsion(t, y):
        return y - 0.5*np.abs(np.sin(3*t))
    
    @staticmethod
    def contempt(t, y):
        return y * (1 - np.abs(np.sin(2*t)))
    
    @staticmethod
    def disdain(t, y):
        return y * (1 - 0.5*np.abs(np.sin(3*t)))

# Protection

class Protection(Emotion):
    def __init__(self, filter_function):
        super().__init__(filter_function)

    def perceive_protection_event(self, event):
        # Implementation for perceiving protection-related events
        pass

    # Function to create a psychic shield modifier
    def psychic_shield(t, y):
        return np.where(np.abs(y) < 0.5, 0, y)
    
    def invoke_crystal_grid(self, grid, t):
        filtered_t = grid.get_waveform(t)
        return filtered_t

    @property
    def state(self):
        return self.emotion

# DSM-5

class DSM(Emotion):
    def __init__(self, filter_function):
        super().__init__(filter_function)
    def perceive_dsm_event(self, event):
      # TBD
      pass
    
    """
    ADHD:

    Rapid fluctuations in attention modeled as abrupt, random filtering windows
    Impulse control issues manifest as spikes and discontinuities
    Hyperactivity captured through overlaid high-frequency sinusoidal fluctuations
    """
    def adhd(y, window_length):
        window_length = np.random.randint(low=5, high=15, size=len(y))
        return savgol_filter(y.ravel(), window_length=window_length, polyorder=2) + 0.1*np.random.randn(len(y)) + np.sin(5*t)

    """
    Autism Spectrum:

    Sensory sensitivity modeled as waveform amplification
    Social difficulties as filtering parts of base signal
    Restricted interests as repetitive waveforms
    Parameterize severity along spectrum
    """
    def autism(y, severity=0.5):
        intensity = 1 + 2*severity 
        social_cutoff = severity*20 # Frequencies filtered 
        repetition = 10 - 7*severity # Periodicity of wave
        
        y_filtered = signal.butter_lowpass_filter(y, social_cutoff, order=4) 
        return np.tile(y_filtered, repetition)[:len(y)] * intensity
    
    """
    Anxiety:

    Baseline unease as slight distortion
    Spiky over-reactivity to stimuli
    Persistent high-frequency rumination
    """

    def anxiety(y):
        baseline = y + 0.1*np.random.uniform(-1, 1, len(y))
        spikes = np.maximum(-5, np.minimum(10*y, 5))  
        rumination = np.sin(10*t)  
        return baseline + spikes + rumination

    """
    Depression:

    Flattened, muted affect
    Occasional dips signaling despair
    Smearing of experience over time
    """

    def depression(y):
        muting = 0.5*np.abs(y)
        dips = np.minimum(-0.5, y) 
        blur = signal.gauss_filter(y, sigma=5)
      
        return  muting + dips + blur

    """
    Other:

    We can in principle model any disorder, trait, symptom list, 
    or mental state from the DSM-V and psychiatry within this paradigm! 
    Here are more code equation examples spanning a diverse range. 
    While rough and speculative, they illustrate the expressive potential:
    """

    # Personality Disorders 
    def borderline_traits(y):
        instability = signal.resample(y, int(len(y)/2))
        intensity = 10*y/(10 + np.abs(y))  
        return 0.6*instability + 0.3*intensity + 0.1*y   

    def antisocial(y):
        no_empathy = np.abs(np.minimum(y, 0))
        risk_taking = 1.5*y* (1 + 0.5*np.random.uniform(-1,1)) 
        return 0.7*no_empathy + 0.3*risk_taking

    # Bipolar / Cyclothymia
    def manic_state(y):
        goal_directed = 2*y* (1 + np.cos(t)) 
        risks = goal_directed + np.random.normal(0, 0.5, len(y)) 
        return 0.7*goal_directed + 0.3*risks

    def depressive_episode(y):
        return depression(y) # Defined previously
      
    # Trauma / Stress / Dissociation  
    def trauma(y):
        fragments = signal.resample(np.abs(y), int(len(y)/2))
        avoidance = np.min([0, y], axis=0)
        flashback = 10*(np.abs(y) - avoidance)  
        return 0.6*avoidance + 0.3*fragments + 0.1*flashback

    # Psychosis continuum 
    def prodromal(y):
        suspicious = np.abs(np.minimum(y, 0))  
        magical = (y+0.5) * (1 + 0.5*np.sin(7*t))
        return 0.7*suspicious + 0.3*magical

    def psychosis(y):  
        delusions = y + np.random.randint(-3, 4, len(y))
        hallucinations = y + np.random.normal(0, 2, len(y)) 
        disorganized = signal.resample(y, int(len(y)/2)) 
        return delusions + hallucinations + disorganized

    # Sensory intensity
    def intensity(x, factor):
        return x * factor

        intense_x = intensity(x, 5)

    # Sensory fragmentation
    def fragmentation(x, n_fragments):
        fragments = np.split(x, n_fragments)
        fragments = np.random.permutation(fragments)
        return np.concatenate(fragments)

        fragments_x = fragmentation(x, 4)

    # Hallucinations 
    def hallucination(x, halluc_scale=0.1):
        hallucinations = np.random.normal(scale=halluc_scale, size=len(x)) 
        return x + hallucinations

        hallucinated_x = hallucination(x)

    # Thought disturbance
    def disturbance(x): 
        X = np.fft.rfft(x)
        phase = np.random.vonmises(mu=0, kappa=20, size=len(X))  
        X_dist = X * np.exp(1j*phase)
        return np.fft.irfft(X_dist)

        disturbed_x = disturbance(x)

    # Suppressed awareness 
    def suppression(x, cutoff=10):
        b, a = signal.butter(3, cutoff, 'lowpass')
        return signal.filtfilt(b, a, x)  

        suppressed_x = suppression(x, 5)

    def emotional_instability(x, severity=0.5):
    
        # Fragmented "agents"
        fragments = signal.resample(x, int(len(x)*(1-severity)))  
        
        # Stochastic spin state transitions  
        perturbations = severity * np.random.randn(len(x))
        
        # Waveform filtering 
        filter_window = np.minimum(15, 50*severity) 
        filtered = signal.savgol_filter(x, window_length=filter_window)  

        return fragments + perturbations + filtered

    def depersonalization(x, num_agents=8, severity=0.5):

        # Splits waveform amongst dissociated self-agents
        agents = np.split(x, num_agents) 
        agents = np.roll(agents, int(20*severity))
        
        # Spin echo disturbance
        echo_delay = int(100*severity)
        spin_echo = signal.hilbert(x, echo_delay)
        
        return sum(agents) + spin_echo*severity

    def racing_thoughts(x, frequency=30, n_threads=8):

        threads = []
        for i in range(n_threads):
            thread = np.sin(frequency*t + i/10)  
            threads.append(thread)

        return sum(threads) + x

    def theory_of_mind(x, level=1):
   
        # Filters waveform to model internal vs. external perspective taking
        b,a = signal.butter(4, level*20, 'low') 
        filtered_x = signal.filtfilt(b,a, x)

        return x - np.abs(filtered_x)

    def hypersensitivity(x, sensitivity=0.5):

        # Models fragile emotional boundaries     
        fragility = np.gradient(np.gradient(x))  

        # Spin probability overload (too much significance perceived)
        overload = sensitivity * (np.abs(x) + 0.1)**2 

        return fragility + overload

    def sensory_sensitivity(x, threshold=0.5):
    
        # Models sensory overload    
        overload = np.clip(threshold * x**2, -1, 1)
        
        # Represent hyper-sensitivity via waveform derivatives
        sensitivity = np.abs(np.gradient(np.gradient(x)))
            
        return overload + sensitivity

    def restrictive_repetitive(x, severity=0.5):

        # Looping, restrictive waveform shape    
        base = np.sin(np.mod(5*t, 2*np.pi)) 
        
        # Additional repetitive high-freq  
        repetitive = severity * np.sin(30 * t)
        
        return base + repetitive

    def attachment_issues(x, severity=0.5):

        # Reduces social signal (human connection)
        social_cutoff_hz = severity*100  
        filtered_x = signal.butter_lowpass_filter(x, social_cutoff_hz, order=6)

        # Models internal working models fearing relationships   
        anxiety = severity * (np.maximum(-1, np.minimum(np.abs(x)-0.3, 1)) ** 2)

        return filtered_x + anxiety
