from enum import Enum
import uuid


class EventType(Enum):
    START_REQUEST = 1
    SCHEDULING = 2
    END_REQUEST = 3
    FINISH_BATCH = 4
    SLO_EXPIRING = 5
    BATCH_EXPIRING = 6


class Event:
    def __init__(self, start_time, type, desc, runtime=None, deadline=1000, id='',
                qos_level=0, accuracy=100.0, predictor=None, executor=None,
                event_counter=0, accuracy_seen=None, late=None):
        self.id = id
        if self.id == '':
            self.id = uuid.uuid4().hex
        self.type = type
        self.start_time = start_time
        self.desc = desc
        self.runtime = runtime
        self.deadline = deadline
        self.qos_level = qos_level
        self.accuracy = accuracy

        # parameters needed for batch processing
        self.predictor = predictor
        self.executor = executor
        # event counter is only set if event is SLO_EXPIRING
        self.event_counter = event_counter
        self.accuracy_seen = accuracy_seen
        self.late = late

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.start_time < other.start_time
        else:
            return NotImplemented

        
class Behavior(Enum):
    BESTEFFORT = 1
    STRICT = 2

class TaskAssignment(Enum):
    RANDOM = 1
    ROUND_ROBIN = 2
    EARLIEST_FINISH_TIME = 3
    LATEST_FINISH_TIME = 4
    INFAAS = 5
    CANARY = 6
    