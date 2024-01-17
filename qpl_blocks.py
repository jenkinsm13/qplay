# qpl_blocks.py

import numpy as np
import random
from datetime import datetime
from typing import ChainMap

# Define cyclic time
t = np.linspace(-np.pi, np.pi, 256)

class ChainError(Exception):
    """Custom exception for errors related to Chain operations."""
    pass

class BlockError(Exception):
    """Custom exception for errors related to Block operations."""
    pass

class AgentError(Exception):
    """Custom exception for errors related to Agent operations."""
    pass

class Chain:
    def __init__(self):
        self.blocks = []
        self.ledger = []
#        self.error_corrector = QuantumErrorCorrector()
#        self.encryption = QuantumSecurity()
#        self.interface = QuantumClassicInterface()

    def add_block(self, block):
        if not isinstance(block, Block):
            raise ChainError("Only Block instances can be added to the chain.")
        if block not in self.blocks:
            self.blocks.append(block)
            self.add_to_ledger(f"Block {block.name} added to the global chain.")

    def add_to_ledger(self, event, source='Global', is_global_event=True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        label = '[Global]' if is_global_event else f'[{source}]'
        entry = (timestamp, label, event) # Create a tuple for the ledger
        self.ledger.append(entry) # Immutable ledger entry
        print(f"{entry}")

    def print_history(self, chain_name='Global'):
        print(f"Chain Ledger ({chain_name}):")
        for entry in self.ledger:
            timestamp, source, event = entry
            print(f"{timestamp} {source}: {event}")

    def __str__(self):
        block_names = ', '.join(block.name for block in self.blocks if block.name)
        return f"Chain(Blocks: {block_names})"

    def __repr__(self):
        return f"Chain(Blocks: {len(self.blocks)})"

# Global immutable chain
global_chain = Chain()

class Block:
    def __init__(self, name, thought_vector=None, state='default'):
        self.name = name
        self.encoded_thought = thought_vector if thought_vector is not None else np.array([])
        self.contained_blocks = {}
        self.contained_agents = {}
        self.connected_blocks = {}
        self.time_state = 0  # Initialize time state
        self.cycle_count = 0
        self.in_use = False  # Flag to indicate if the block is in use

        # Each block has its own chain
        self.chain = Chain()
        self.chain.add_block(self)

        # Event in the block's own chain
        self.chain.add_to_ledger(f"Chain for {self.name} created.", source=self.name, is_global_event=False)

        # Only update global chain if it's not an instance of a subclass
        global_chain.add_to_ledger(f"Block {self.name} initialized.", source=self.name, is_global_event=False)

    def update_time_state(self):
        if self.in_use:
            # Update time state cyclically
            new_time_state = (self.time_state + np.pi / 2) % (2 * np.pi)
            if new_time_state < self.time_state:
                # A cycle is completed
                self.cycle_count += 1
            self.time_state = new_time_state

    def set_in_use(self, in_use=True):
        self.in_use = in_use

    def add_block(self, block):
        if not isinstance(block, Block):
            raise BlockError("Only Block instances can be added.")
        self.contained_blocks.append(block)
        block.chain = self.chain
        if self.chain:
            self.chain.add_to_ledger(f"Block {block.name} added to {self.name}.")

    def add_agent(self, agent):
        if not isinstance(agent, Agent):
            raise BlockError("Only Agent instances can be added.")
        self.contained_agents.append(agent)
        agent.chain = self.chain
        if self.chain:
            self.chain.add_to_ledger(f"Agent {agent.name} added to {self.name}.")

    def connect(self, other_block):
        if not isinstance(other_block, Block):
            raise BlockError("Only Block instances can be connected.")
        if other_block not in self.connected_blocks:
            self.connected_blocks.append(other_block)
            other_block.connected_blocks.append(self)
            if self.chain:
                self.chain.add_to_ledger(f"Block {other_block.name} connected to {self.name}.")

    @property
    def state(self):
        return self.encoded_thought  # Placeholder for actual thought decoding

    def entangle_with(self, other_block):
        if not isinstance(other_block, Block):
            raise BlockError("Entanglement requires another Block.")
        # Placeholder for quantum entanglement logic
        self.chain.add_to_ledger(f"Block {self.name} entangled with {other_block.name}")

    def __str__(self):
        return f"Block({self.name}, thought: {self.encoded_thought})"

class HyperBlock(Block):
    def __init__(self, name, thought_vector, energy_level, sub_blocks=[], state='superposition'):
        super().__init__(name, thought_vector=None, state=state)
        self.knowledge_base = {}
        self.energy_level = energy_level
        self.sub_blocks = sub_blocks

    def activate(self):
        print(f"HyperBlock {self.name} activated at energy level {self.energy_level}")

    def learn(self, block):
        if not isinstance(block, Block):
            raise AgentError("Only Block instances can be added to knowledge.")
        self.knowledge_base[block.name] = block
        if self.chain:
            self.chain.add_to_ledger(f"{block.name} added to knowledge of {self.name}.", source=self.name, is_global_event=False)

    def traverse_and_learn(self, start_block):
            visited = {}
            stack = [start_block]

            while stack:
              block = stack.pop()
              if block not in visited:
                  self.learn(block)
                  visited[block] = True
                  for connected in block.connected_blocks:
                      stack.append(connected)

class Agent(HyperBlock):
    def __init__(self, name, thought_vector=None, energy_level=0, state='active'):
        # Initialize as Block but don't add to global chain yet
        super().__init__(name, thought_vector, energy_level, state)
        self.tool_box = {}
        self.active_block = None
        # Update global chain for agent creation
        self.chain.add_to_ledger(f"Agent {self.name} created.", source=self.name, is_global_event=False)

    def emerge_new_block(self):
       new_block = QuantumBlock()
       for known in self.knowledge_base:
           new_block.connect(known)
       return new_block

    def pick_up_block(self, block):
        if not isinstance(block, Block):
            raise AgentError("Only Block instances can be picked up.")
        if block.name in self.tool_box:
            return
        self.tool_box[block.name] = block
        self.chain.add_to_ledger(f"Block {block.name} picked up by Agent {self.name}.", source=self.name, is_global_event=False)
        global_chain.add_to_ledger(f"Agent {self.name} picked up Block {block.name}.", source=self.name, is_global_event=False)

    def put_down_block(self, block):
        if block.name not in self.tool_box:
            return
        del self.tool_box[block.name]
        self.chain.add_to_ledger(f"Block {block.name} put down by Agent {self.name}.", source=self.name, is_global_event=False)
        global_chain.add_to_ledger(f"Agent {self.name} put down Block {block.name}.", source=self.name, is_global_event=False)

    def move_to_block(self, block):
        if block.name not in self.tool_box:
            raise AgentError("Can only move to a block that is in tools.")
        self.active_block = block
        self.chain.add_to_ledger(f"Block {block.name} moved to by Agent {self.name}.", source=self.name, is_global_event=False)
        global_chain.add_to_ledger(f"Agent {self.name} moved to Block {block.name}.", source=self.name, is_global_event=False)

    def perform_operation(self, operation):
        # Logic for performing an operation
        # Example: Update the state, interact with blocks, etc.
        # This is just an example. You should replace it with actual logic.
        self.current_operation = operation
        print(f"Agent {self.name} is performing operation: {operation}")

        # Add to ledger
        global_chain.add_to_ledger(f"Agent {self.name} performed operation: {operation}", source=self.name, is_global_event=True)
        self.chain.add_to_ledger(f"{operation} performed by {self.name}", source=self.name, is_global_event=True)

        # Return some result or state change if needed
        return f"Operation {operation} performed"

    def use_tool(self, tool_name, parameters):
        """Use tool from toolkit
        Example: y, t = agent.use_tool("Joy", {"y": y_wave, "t": t_wave})"""
        if tool_name not in self.toolbox:
            raise AgentError("Tool not acquired.")
        tool = self.tool_box[tool_name]
        global_chain.add_to_ledger(f"Agent {self.name} used tool: {tool}", source=self.name, is_global_event=True)
        y = params["y"]  
        t = params["t"]
        # Apply tool/filter 
        y, t = tool.apply(y, t)  
        # Handle visualization updates
        update_waveform_plot(y, t)
        return y, t 


    def ponder(self, question):
        """Think with current knowledge to yield insights"""  
        return assimilate(question, self.knowledge_base)

    def perceive(self, quantum_block):
        # Process quantum data
        perception = f"Processing {quantum_block.state}"
        self.knowledge_base['last_perception'] = perception
        self.chain.add_to_ledger(f"Agent {self.name} perceived {quantum_block.name}.", source=self.name, is_global_event=False)
        return perception

    def decide(self, action_options):
        # Make a decision based on quantum probability
        decision = random.choice(action_options)
        self.chain.add_to_ledger(f"Agent {self.name} decided to {decision}.", source=self.name, is_global_event=False)
        return decision

    def recollect(self):
        # Recollect knowledge from memory
        self.chain.add_to_ledger(f"Agent {self.name} recollected knowledge.", source=self.name, is_global_event=False)
        return self.knowledge_base

    def __str__(self):
        return f"Agent({self.name}, thought: {self.encoded_thought}, knowledge: {self.knowledge})"

class HyperAgent(Agent):
    def __init__(self, name):
        super().__init__(name, 100000)
        self.strategies = []

    def manage_block(self, block):
        # HyperAgent executes strategy on Block
        pass

    def decide(self, action_options):
        """Make a decision based on quantum probability."""
        decision = random.choice(action_options)
        return decision

    def manage_block(self, block):
        """HyperAgent executes strategy on Block."""
        # Placeholder for strategy execution logic

    def quantum_loop(condition, quantum_block):
        """Quantum loop control structure."""
        while condition(quantum_block):
            yield quantum_block.state
