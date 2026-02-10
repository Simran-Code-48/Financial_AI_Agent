import pandas as pd

def normalize_amounts(df):
    df["AMOUNT"] = 0.0

    credit_mask = df["DEPOSITS"].fillna(0) > 0
    debit_mask = df["WITHDRAWALS"].fillna(0) > 0

    df.loc[credit_mask, "AMOUNT"] = df.loc[credit_mask, "DEPOSITS"]
    df.loc[debit_mask, "AMOUNT"] = df.loc[debit_mask, "WITHDRAWALS"]

    df["DIRECTION"] = None
    df.loc[credit_mask, "DIRECTION"] = "CREDIT"
    df.loc[debit_mask, "DIRECTION"] = "DEBIT"

    return df