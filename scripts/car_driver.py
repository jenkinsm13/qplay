import blocks as qpl
from blocks import BlockError, AgentError

class Car(qpl.Block):
    def __init__(self, name, thought_vector=None):
        super().__init__(name, thought_vector)
        self.is_driven = False
        self.current_driver = None

    def start_engine(self, driver):
        if not isinstance(driver, Driver):
            raise TypeError("Only a Driver can start the car.")
        if self.is_driven:
            raise BlockError(f"Car {self.name} cannot be started. It is already being driven by {self.current_driver.name}.")

        self.is_driven = True
        self.current_driver = driver
        self.chain.add_to_ledger(f"Car {self.name} was started by {driver.name}.")

    def stop_engine(self, driver):
        if driver != self.current_driver:
            raise BlockError(f"Car {self.name} can only be stopped by {self.current_driver.name}.")
        self.is_driven = False
        self.current_driver = None
        self.chain.add_to_ledger(f"Car {self.name} has been stopped by {driver.name}.")

class Driver(qpl.Agent):
    def __init__(self, name, thought_vector=None):
        super().__init__(name, thought_vector)

    def start_car(self, car):
        if not isinstance(car, Car):
            raise AgentError("The driver can only drive a Car.")
        car.start_engine(self)
        self.chain.add_to_ledger(f"Driver {self.name} started {car.name}.")

    def stop_car(self, car):
        if not isinstance(car, Car):
            raise AgentError("The driver can only stop a Car.")
        car.stop_engine(self)
        self.chain.add_to_ledger(f"Driver {self.name} stopped the car {car.name}.")
