# /utils/qplay_gui.py

import ipywidgets as widgets
import inspect
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from blocks import *
from IPython.display import display, clear_output
from mpl_toolkits.mplot3d import Axes3D



class GUI(Block):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.gui_ledger = []

    def create_game_boy_ui(self):
        # Define button style
        button_style = {'description_width': '0px', 'button_width': '50px', 'button_height': '50px'}

        # Define buttons
        up_button = widgets.Button(description='U', layout=widgets.Layout(width=button_style['button_width'], height=button_style['button_height']))
        down_button = widgets.Button(description='D', layout=widgets.Layout(width=button_style['button_width'], height=button_style['button_height']))
        left_button = widgets.Button(description='L', layout=widgets.Layout(width=button_style['button_width'], height=button_style['button_height']))
        right_button = widgets.Button(description='R', layout=widgets.Layout(width=button_style['button_width'], height=button_style['button_height']))
        a_button = widgets.Button(description='A', layout=widgets.Layout(width=button_style['button_width'], height=button_style['button_height']))
        b_button = widgets.Button(description='B', layout=widgets.Layout(width=button_style['button_width'], height=button_style['button_height']))
        start_button = widgets.Button(description='+', layout=widgets.Layout(width=button_style['button_width'], height=button_style['button_height']))
        select_button = widgets.Button(description='-', layout=widgets.Layout(width=button_style['button_width'], height=button_style['button_height']))
        home_button = widgets.Button(description='🏠', layout=widgets.Layout(width=button_style['button_width'], height=button_style['button_height']))

        # Define game ledger
        self.game_boy_ledger = []

        # Create a 3x3 grid layout for the d-pad
        d_pad_layout = widgets.GridspecLayout(2, 3, layout=widgets.Layout(align_items='center', justify_content='center'))
        d_pad_layout[0, 1] = up_button
        d_pad_layout[1, 0] = left_button
        d_pad_layout[1, 2] = right_button
        d_pad_layout[1, 1] = down_button

        # Define layout for home, start, select buttons
        center_box = widgets.GridspecLayout(2, 3, layout=widgets.Layout(align_items='center', justify_content='center'))
        center_box[0, 1] = home_button
        center_box[0, 0] = select_button
        center_box[0, 2] = start_button

        # Define layout for A, B buttons
        right_box = widgets.GridspecLayout(2, 2, layout=widgets.Layout(align_items='center', justify_content='center'))
        right_box[0, 1] = a_button
        right_box[1, 0] = b_button

        # Create a separate output widget for the ledger
        self.ledger_output = widgets.Output()

        # Create UI layout
        # Define layout
        ui = widgets.HBox([d_pad_layout, center_box, right_box])

        # Attach event handlers
        for button in [up_button, down_button, left_button, right_button, a_button, b_button, home_button, start_button, select_button]:
            button.on_click(self.handle_game_boy_press)

        return widgets.VBox([ui, self.ledger_output])

    def handle_game_boy_press(self, button):
        # Log the button press
        self.game_boy_ledger.append(f"Button pressed: {button.description}")

        # Update the ledger display
        with self.ledger_output:
            clear_output(wait=True)
            print("Game Boy Ledger:")
            for event in self.game_boy_ledger:
                print(event)

    @staticmethod
    def create_filter_selection_widget(target_class):
        # Retrieve all methods of the class
        methods = inspect.getmembers(target_class, predicate=inspect.isfunction)

        # Filter methods that end with '_filter'
        filter_methods = [name for name, _ in methods if name.endswith('_filter')]

        # Format method names: remove '_filter' and convert to sentence case
        formatted_names = [name.replace('_filter', '').replace('_', ' ').title() for name in filter_methods]

        # Get the class name
        class_name = target_class.__name__

        # Create a multi-select list widget
        list_widget = widgets.SelectMultiple(
            options=formatted_names,
            description=f'{class_name}:',
            disabled=False
        )

        return list_widget

    @staticmethod
    def create_widgets_for_subclasses(target_class):
        widgets_list = []
        for subclass in target_class.__subclasses__():
            widget = GUI.create_filter_selection_widget(subclass)
            widgets_list.append(widget)
        return widgets_list

    @staticmethod
    def display_emotion_filters(target_class):
        # Create widgets for subclasses
        emotion_filters = GUI.create_widgets_for_subclasses(target_class)

        # Create a GridspecLayout with 4 rows and 3 columns
        emotion_grid = widgets.GridspecLayout(4, 3)

        # Add widgets to the grid
        for i, widget in enumerate(emotion_filters):
            row = i // 3  # Determine row
            col = i % 3   # Determine column
            emotion_grid[row, col] = widget

        # Display the grid
        display(emotion_grid)


class GraphVisualizer(GUI):
    """
    GraphVisualizer class for visualizing 3D graphs.
    """
    def __init__(self):
        self.t = np.linspace(-np.pi, np.pi, 256)
        self.fig = plt.figure(figsize=(6,6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.output = widgets.Output()
        self.reset_graph()
        self.initial_plot()

    def init(self):
        self.line.set_data([], [])
        self.line.set_3d_properties([])
        return self.line

    def initial_plot(self):
        with self.output:
            self.plot_graph()

    def plot_graph(self):
        with self.output:
            clear_output(wait=True)
            self.ax.clear()
            self.ax.plot(self.x, self.y, self.z)
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.ax.set_zlim(-1, 1)
            plt.show()

    def reset_graph(self):
        self.x = np.sin(self.t)
        self.y = np.sin(2 * self.t) / 2
        self.z = np.cos(self.t)
        self.plot_graph()

    def rotate_graph(self, angle, axis):
        if axis == 'x':
            self.y, self.z = self.y * np.cos(angle) - self.z * np.sin(angle), self.y * np.sin(angle) + self.z * np.cos(angle)
        elif axis == 'y':
            self.x, self.z = self.x * np.cos(angle) - self.z * np.sin(angle), self.x * np.sin(angle) + self.z * np.cos(angle)
        elif axis == 'z':
            self.x, self.y = self.x * np.cos(angle) - self.y * np.sin(angle), self.x * np.sin(angle) + self.y * np.cos(angle)
        self.plot_graph()

    def rotate_view(self, angle, axis):
        if axis == 'vertical':
            self.ax.view_init(elev=self.ax.elev + angle)
        elif axis == 'horizontal':
            self.ax.view_init(azim=self.ax.azim + angle)
        self.fig.canvas.draw_idle()

    def rotate_left_action(self, button):
        self.rotate_graph(np.pi/8, 'y')

    def rotate_right_action(self, button):
        self.rotate_graph(-np.pi/8, 'y')

    def reset_graph_action(self, button):
        self.reset_graph()

    def rotate_view_left(self, button):
        self.rotate_view(-15, 'horizontal')

    def rotate_view_right(self, button):
        self.rotate_view(15, 'horizontal')

    def rotate_view_up(self, button):
        self.rotate_view(-15, 'vertical')

    def rotate_view_down(self, button):
        self.rotate_view(15, 'vertical')

    def reset_view(self, button):
        self.ax.view_init(elev=None, azim=None)
        self.fig.canvas.draw_idle()

    def zoom_in(self, button):
        min_dist = 3
        if self.ax.dist > min_dist:
            self.ax.dist -= 1
            self.fig.canvas.draw_idle()

    def zoom_out(self, button):
        min_dist = 3
        if self.ax.dist > min_dist:
            self.ax.dist += 1
            self.fig.canvas.draw_idle()

    def apply_emotion_filter(self, filter_function):
        # filters is a list of selected filter functions
        for filter_function in filters:
            self.y = filter_function(self.y, self.t)
        self.plot_graph()

    def on_emotion_filter_change(self, change):
        # Get the selected emotion names from the change event
        selected_emotions = change['new']

        # Apply each selected emotion filter
        for emotion_name in selected_emotions:
            # Look up the filter function by emotion name
            for emotion_class in [Positive, Negative, Protection]:
                try:
                    filter_func = getattr(emotion_class, f"{emotion_name}_filter")
                    self.apply_emotion_filter(filter_func)
                    break
                except AttributeError:
                    continue

    def display_combined_ui(self, target_class):
        
        # Add actions to the GUI buttons
        gui = GUI("Example GUI")
        game_boy_ui_with_ledger = self.create_game_boy_ui()

        # Example: Connect the left and right buttons
        left_button = game_boy_ui_with_ledger.children[0].children[0].children[1]  # Accessing the left button
        left_button.on_click(self.rotate_view_left)

        right_button = game_boy_ui_with_ledger.children[0].children[0].children[2]  # Accessing the right button
        right_button.on_click(self.rotate_view_right)

        up_button = game_boy_ui_with_ledger.children[0].children[0].children[0]  # Accessing the left button
        up_button.on_click(self.rotate_view_up)

        down_button = game_boy_ui_with_ledger.children[0].children[0].children[3]  # Accessing the right button
        down_button.on_click(self.rotate_view_down)

        home_button = game_boy_ui_with_ledger.children[0].children[1].children[0]  # Accessing the home button
        home_button.on_click(self.reset_graph_action)

        a_button = game_boy_ui_with_ledger.children[0].children[2].children[0]  # Accessing the A button
        a_button.on_click(self.rotate_left_action)

        b_button = game_boy_ui_with_ledger.children[0].children[2].children[1]  # Accessing the A button
        b_button.on_click(self.rotate_right_action)

        minus_button = game_boy_ui_with_ledger.children[0].children[1].children[1]  # Accessing the minus button
        minus_button.on_click(self.zoom_out)

        plus_button = game_boy_ui_with_ledger.children[0].children[1].children[2]  # Accessing the minus button
        plus_button.on_click(self.zoom_in)


        # Create emotion filter grid
        emotion_filters = self.create_widgets_for_subclasses(target_class)
        emotion_grid = widgets.GridspecLayout(4, 3)
        for i, widget in enumerate(emotion_filters):
            row = i // 3  # Determine row
            col = i % 3   # Determine column
            emotion_grid[row, col] = widget
            # Set up observation for each widget
            widget.observe(self.on_emotion_filter_change, names='value')

        # Display the visualizer, game boy UI, and emotion filters grid
        display(self.output)
        display(game_boy_ui_with_ledger)
        display(emotion_grid)
