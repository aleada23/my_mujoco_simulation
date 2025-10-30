import os

class Environment:
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Environment file not found: {path}")
        self.path = path

    def get_path(self):
        return self.path
