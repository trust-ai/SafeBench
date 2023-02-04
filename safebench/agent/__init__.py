from .dummy_agent.dummy import DummyEgo
from .object_detection.example import ObjectDetection
AGENT_LIST = {
    'dummy': DummyEgo,
    'object_detection': ObjectDetection,
}
