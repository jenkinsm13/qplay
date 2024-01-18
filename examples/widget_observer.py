import time
import blocks
from blocks import global_chain

example_block = Block(name='Widget', state='entangled')
time.sleep(2)  # Pauses execution

example_agent = Agent(name='Observer')
time.sleep(2)  # Pauses execution

perception = example_agent.perceive(example_block)
time.sleep(2)  # Pauses execution

decision = example_agent.decide(['explore', 'observe', 'analyze'])
time.sleep(2)  # Pauses execution

if decision == 'explore':
    example_agent.perform_operation('explore')
elif decision == 'observe':
    example_agent.learn(example_block)
else :
    example_agent.pick_up_block(example_block)

time.sleep(2)  # Pauses execution

print(f"Agent's Perception: {perception}")
print(f"Agent's Decision: {decision}")
global_chain.print_history()