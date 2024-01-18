# /blocks/qplay_movement.py

import math
import numpy as np
import qplay_blocks
from qplay_blocks import Block, Agent

class Movement(Block):
    """
    Base class for movement-related functionalities in the QPL system.
    Inherits from Block and represents a fundamental movement action.
    """
    def __init__(self, name, chain, thought_vector=None):
        super().__init__(name, chain, thought_vector)
        # Additional initialization specific to movement

class Linear(Movement):
    """
    Represents linear movement along a specific axis.
    Inherits from Movement.
    """
    def __init__(self, name, chain, axis, distance, thought_vector=None):
        super().__init__(name, chain, thought_vector)
        self.axis = axis
        self.distance = distance

    def move(self, agent):
        """
        Executes linear movement for the given agent.

        Args:
            agent (Agent): The agent to be moved.
        """
        if self.axis == 'x':
            agent.position['x'] += self.distance
        elif self.axis == 'y':
            agent.position['y'] += self.distance
        # Update history to log the movement
        self.update_history(f"Moved {self.distance} units along the {self.axis} axis.")

class Radial(Movement):
    """
    Represents rotational movement around a specific axis.
    Inherits from Movement.
    """
    def __init__(self, name, chain, axis, angle, thought_vector=None):
        super().__init__(name, chain, thought_vector)
        self.axis = axis
        self.angle = angle  # Angle in radians

    def rotate(self, agent):
        """
        Executes rotational movement for the given agent.

        Args:
            agent (Agent): The agent to be rotated.
        """
        # Here you would implement the logic to rotate the agent's orientation
        # This is a placeholder for your system's specific implementation
        # For example, you might apply a rotation matrix to the agent's orientation
        if self.axis == 'z':
            agent.orientation = self.rotate_z(agent.orientation, self.angle)
        self.update_history(f"Rotated {math.degrees(self.angle)} degrees around the {self.axis} axis.")
