
class ConfigException(Exception):
    """Exception raised for errors in experiment configuration file.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class SimulatorException(Exception):
    """Exception raised for errors in the Simulator.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ExecutorException(SimulatorException):
    """Exception raised for errors in the Executor.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class PredictorException(SimulatorException):
    """Exception raised for errors in the Predictor.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class IlpException(Exception):
    """Exception raised for errors in the ILP.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
        