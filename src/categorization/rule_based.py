from src.config import CATEGORY_OTHER, SUBCATEGORY_UNCATEGORIZED

FOOD_KEYWORDS = [
    "zomato", "swiggy", "domino", "pizza", "restaurant", "hotel", "cafe"
]

TRANSPORT_KEYWORDS = [
    "uber", "ola", "rapido", "metro", "bus", "irctc", "rail"
]

SHOPPING_KEYWORDS = [
    "amazon", "flipkart", "myntra", "ajio"
]

BILLS_KEYWORDS = [
    "electric", "electricity", "airtel", "jio", "vi",
    "broadband", "fiber", "gas", "water"
]

SUBSCRIPTION_KEYWORDS = [
    "netflix", "spotify", "prime", "hotstar", "youtube"
]

def normalize_text(*values):
    return " ".join(
        str(v).lower()
        for v in values
        if v and isinstance(v, str)
    )

def rule_based_category(row):
    text = normalize_text(
        row.get("COUNTERPARTY_NAME"),
        row.get("COUNTERPARTY_VPA"),
        row.get("REMARKS")
    )

    # Defaults
    category = CATEGORY_OTHER
    subcategory = SUBCATEGORY_UNCATEGORIZED
    confidence = 0.3

    # --- HARD RULES (highest confidence) ---
    if row["TXN_TYPE"] == "SALARY" and row["DIRECTION"] == "CREDIT":
        return "INCOME", "SALARY", "RULE", 0.95

    if row["TXN_TYPE"] == "CASH_WITHDRAWAL":
        return "CASH", "CARDLESS", "RULE", 0.99

    # --- KEYWORD RULES ---
    if any(k in text for k in FOOD_KEYWORDS):
        return "FOOD", "ONLINE_DELIVERY", "RULE", 0.85

    if any(k in text for k in TRANSPORT_KEYWORDS):
        return "TRANSPORT", "CAB", "RULE", 0.8

    if any(k in text for k in SHOPPING_KEYWORDS):
        return "SHOPPING", "ONLINE", "RULE", 0.8

    if any(k in text for k in BILLS_KEYWORDS):
        return "BILLS", "UTILITY", "RULE", 0.85

    if any(k in text for k in SUBSCRIPTION_KEYWORDS):
        return "SUBSCRIPTION", "OTT", "RULE", 0.85

    # --- TRANSFER HEURISTIC ---
    if row["TXN_TYPE"] == "UPI" and row["DIRECTION"] == "DEBIT":
        return "TRANSFER", "PERSON", "RULE", 0.6

    return category, subcategory, "RULE", confidence