# /toys/qplay_toys.py

import qplay_blocks
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import qplay_emotions
import inspect

class QuantumToy(Block):
    def __init__(self, name):
        self.name = name

    def activate(self):
        print(f"{self.name} is now active.")

    def deactivate(self):
        print(f"{self.name} is now inactive.")

    def get_categories():
        categories = []
        for name, cls in inspect.getmembers(qpl_emotions, inspect.isclass):
            if name.endswith("Emotions"):
                categories.append(name[:-8])
        return categories

class QuantumToyGUI(Block):
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.dropdowns = {}
        self.update_button = widgets.Button(description="Update Visualization")
        self.update_button.on_click(self.on_update_button_clicked)
        self.create_dropdowns()
        self.display()

    def create_dropdowns(self):
        categories = get_categories()
        for category in categories:
            dropdown = widgets.SelectMultiple()
            # Add emotions for this category
            for cls in get_category_emotions(category):
                name = get_name(cls)
                dropdown.options.append(name)
            self.dropdowns[category] = dropdown

    def on_update_button_clicked(self, b):
        # Collect selected emotions from all dropdowns
        selected_emotions = set()
        for dropdown in self.dropdowns.values():
            selected_emotions.update(dropdown.value)
        self.visualizer.selected_emotions = selected_emotions
        self.visualizer.visualize_emotions()

    def display(self):
        vbox = widgets.VBox()
        for key, dropdown in self.dropdowns.items():
            hbox = widgets.HBox([widgets.Label(key), dropdown])
            vbox.children += (hbox,)

        display(vbox)
        display(self.update_button)

class EmotionVisualizer(QuantumToy):
    def __init__(self, name, emotions_by_category):
        super().__init__(name)
        self.base_freq = 42
        self.t = np.linspace(0, 2 * np.pi, 100)
        self.x = np.sin(self.t)
        self.y = np.sin(2 * self.t) / 2
        self.z = np.cos(self.t)
        self.emotions_by_category = emotions_by_category
        self.selected_emotions = set()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.initial_plot()

    def initial_plot(self):
        self.ax.plot(self.x, self.y, self.z)
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        plt.show()

    def _lookup_category(self, name):
        # Method logic
        for category, emotions in self.emotions_by_category.items():
            if name in emotions:
                 return category

        return None

    def visualize_emotions(self):
        y_emotion = self.y.copy()
        for name in self.selected_emotions:
            # Add prefix
            func_name = f"{name}"
            # Get category for this emotion
            category = self._lookup_category(name)
            # Get category emotions
            emotions = self.emotions_by_category[category]
            # Lookup in that category
            emotion = emotions[func_name]
            # Call function
            y_emotion = emotion(y_emotion, t=self.t)

        # Clear the previous plot and plot the new waveform
        self.ax.clear()
        self.ax.plot(self.x, y_emotion, self.z)
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        plt.draw()  # Update the plot
        plt.show()

# Function to get emotions by category
def get_emotions_by_category():
    emotions_by_category = {}
    for name, cls in inspect.getmembers(qpl_emotions, inspect.isclass):
        category_name = type(cls).__name__.replace("Emotions", "")
        emotions_by_category[category_name] = {}
    for name, func in inspect.getmembers(cls, inspect.isfunction):
        if name.startswith('emotion'):
            name = name[len(''):]
            emotions_by_category[category_name][name] = func
    return emotions_by_category

#Example usage
emotions_by_category = get_emotions_by_category()
visualizer = EmotionVisualizer("EmotionVisualizer", emotions_by_category)
gui = QuantumToyGUI(visualizer)
