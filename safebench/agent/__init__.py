from .dummy_agent.dummy import DummyEgo
from .object_detection.detector import ObjectDetection

AGENT_LIST = {
    'dummy': DummyEgo,
    'object_detection': ObjectDetection,
}
