from enum import Enum

class UpdateFrequency(Enum):
    COMPLETED = "completed"
    REAL_TIME = "real-time"
    DELAYED = "delayed"
    OTHER = "other"
    UNKNOWN = "unknown"
    BOTH = "both"