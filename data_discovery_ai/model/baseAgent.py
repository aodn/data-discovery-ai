class BaseAgent:
    def __init__(self):
        self.type = "base"
        self.id = None
        # 0 as inactivate, 1 as active, 2 as finished
        self.status = 0
        self.response = {}

    def set_status(self, status: int):
        self.status = status