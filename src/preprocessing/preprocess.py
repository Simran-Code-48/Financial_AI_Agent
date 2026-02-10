import pandas as pd
from .parse_particulars import parse_icici_particulars
from .normalize_amounts import normalize_amounts
from src.config import RAW_DATA_PATH, PREPROCESSED_PATH

def preprocess():
    df = pd.read_csv(RAW_DATA_PATH)

    improved_df = df.copy()
    improved_df["PARTICULARS_RAW"] = improved_df["PARTICULARS"]
    improved_df = improved_df.drop(columns=["PARTICULARS"])

    parsed = improved_df.apply(parse_icici_particulars, axis=1)
    improved_df = pd.concat([improved_df, parsed], axis=1)

    improved_df = normalize_amounts(improved_df)

    improved_df.to_csv(PREPROCESSED_PATH, index=False)
    print(f"Preprocessing complete. Saved to {PREPROCESSED_PATH}")
    
    return improved_df

if __name__ == "__main__":
    preprocess()