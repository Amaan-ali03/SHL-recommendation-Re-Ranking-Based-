import pandas as pd
from src.config import DATASET_PATH

def main():
    train = pd.read_excel(DATASET_PATH, sheet_name="Train-Set")
    test  = pd.read_excel(DATASET_PATH, sheet_name="Test-Set")

    print("Train shape:", train.shape)
    print("Train columns:", list(train.columns))
    print("Test shape:", test.shape)
    print("Test columns:", list(test.columns))

    assert "Query" in train.columns and "Assessment_url" in train.columns, "Train-Set columns mismatch"
    assert "Query" in test.columns, "Test-Set columns mismatch"

    print(" Dataset looks good.")

if __name__ == '__main__':
    main()
