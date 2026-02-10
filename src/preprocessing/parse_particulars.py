import pandas as pd

def parse_icici_particulars(row):
    text = row["PARTICULARS_RAW"]
    result = {
        "TXN_TYPE": "OTHER",
        "COUNTERPARTY_NAME": None,
        "COUNTERPARTY_VPA": None,
        "UTR": None,
        "REMARKS": None,
        "CARD_LAST4": None,
        "CHANNEL": None,
    }

    if text.startswith("UPI"):
        parts = text.split("/")
        result["TXN_TYPE"] = "UPI"

        # Name
        if len(parts) > 1:
            result["COUNTERPARTY_NAME"] = parts[1].strip()

        # VPA
        for p in parts:
            if "@" in p:
                result["COUNTERPARTY_VPA"] = p.strip()
                break

        # UTR
        for p in parts:
            if p.startswith("UPI") and len(p) > 10:
                result["UTR"] = p.strip()[3:]
                break

        # Remarks
        if len(parts) > 3:
            result["REMARKS"] = parts[3]

    elif text.startswith("CMS"):
        parts = [p.strip() for p in text.split("/") if p.strip()]
        result["TXN_TYPE"] = "CMS"

        if len(parts) >= 2:
            result["REMARKS"] = parts[1]

        if len(parts) >= 3:
            result["COUNTERPARTY_NAME"] = parts[-1]

    elif text.startswith("CCW"):
        parts = [p.strip() for p in text.split("/") if p.strip()]
        result["TXN_TYPE"] = "CASH_WITHDRAWAL"

        if len(parts) >= 4:
            result["CARD_LAST4"] = parts[3]

        if len(parts) >= 1:
            result["REMARKS"] = "Cardless Cash WDL"

        if len(parts) >= 6:
            result["CHANNEL"] = parts[-1]
    else:
        pass

    return pd.Series(result)