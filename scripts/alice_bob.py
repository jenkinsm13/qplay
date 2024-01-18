import random
import time
from blocks import global_chain, BlockError, AgentError
from car_driver import Car, Driver

# Create some drivers
alice = Driver("Alice")
time.sleep(2)  # Pauses execution for 5 seconds

bob = Driver("Bob")
time.sleep(2)  # Pauses execution for 5 seconds

# Create some cars
car1 = Car("Tesla Model S")
time.sleep(2)  # Pauses execution for 5 seconds

car2 = Car("Toyota Prius")
time.sleep(2)  # Pauses execution for 5 seconds

# Alice starts car1
alice.start_car(car1)
time.sleep(2)  # Pauses execution for 5 seconds


# Bob tries to start car1 but fails since Alice is already driving it
try:
    bob.start_car(car1)
    time.sleep(2)  # Pauses execution for 5 seconds

except BlockError as e:
    print(e)
    time.sleep(2)  # Pauses execution for 5 seconds


# Bob starts car2 instead
bob.start_car(car2)
time.sleep(2)  # Pauses execution for 5 seconds


# Alice stops car1, freeing it up for another driver
alice.stop_car(car1)
time.sleep(2)  # Pauses execution for 5 seconds


# Now Bob can start car1
bob.start_car(car1)
time.sleep(2)  # Pauses execution for 5 seconds


# Print history
global_chain.print_history()
alice.chain.print_history("Alice")
bob.chain.print_history("Bob")
car1.chain.print_history("Tesla Model S")
car2.chain.print_history("Toyota Prius")