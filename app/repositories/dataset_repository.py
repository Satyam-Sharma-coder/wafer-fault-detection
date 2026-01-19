import pandas as pd

class DatasetRepository:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path)
        return df
