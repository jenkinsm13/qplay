import time
import blocks as qpl
from blocks import global_chain

idea1 = qpl.Block("Piano")
time.sleep(2)  # Pauses execution

idea2 = qpl.Block("Man")
time.sleep(1)  # Pauses execution

agent = qpl.Agent("Billy Joel")
time.sleep(3)  # Pauses execution

agent.learn(idea1)
time.sleep(2)  # Pauses execution

agent.learn(idea2)
time.sleep(1)  # Pauses execution

idea1.entangle_with(idea2)
time.sleep(2)  # Pauses execution

agent.perform_operation("Sing")
time.sleep(3)  # Pauses execution

global_chain.print_history()
idea1.chain.print_history(chain_name=idea1.name)
idea2.chain.print_history(chain_name=idea2.name)
agent.chain.print_history(chain_name=agent.name)