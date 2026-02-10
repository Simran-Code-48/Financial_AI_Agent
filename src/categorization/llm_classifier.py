import json
import time
# import google.generativeai as genai
import cohere
from src.config import GOOGLE_API_KEY, COHERE_API_KEY, LLM_THRESHOLD

# Configure APIs
# genai.configure(api_key=GOOGLE_API_KEY)
co = cohere.Client(COHERE_API_KEY)

CATEGORIES = [
    "INCOME", "TRANSFER", "FOOD", "TRANSPORT", "SHOPPING",
    "BILLS", "ENTERTAINMENT", "HEALTH", "EDUCATION",
    "SUBSCRIPTION", "CASH", "RENT", "INVESTMENT",
    "LOAN", "TAX", "DONATION", "OTHER"
]

SUBCATEGORIES = {
    "INCOME": ["SALARY", "BONUS", "REFUND", "INTEREST", "CASHBACK", "OTHER_INCOME"],
    "FOOD": ["RESTAURANT", "ONLINE_DELIVERY", "GROCERIES", "CAFE"],
    "TRANSFER": ["PERSON", "FAMILY", "FRIEND", "SELF"],
    "CASH": ["ATM", "CARDLESS"],
    "SUBSCRIPTION": ["OTT", "SOFTWARE", "MUSIC"],
    "OTHER": ["UNCATEGORIZED"]
}

# Batch prompt template
BATCH_PROMPT_TEMPLATE = """
You are a financial transaction classifier. Classify each transaction in the list.

RULES:
1. For each transaction, choose CATEGORY only from this list:
   {CATEGORIES}

2. Choose SUBCATEGORY only from the allowed subcategories for that CATEGORY:
   {SUBCATEGORIES}

3. For each transaction, output MUST include:
   - category (string): from CATEGORIES list
   - subcategory (string): from corresponding SUBCATEGORIES list
   - confidence (float): between 0.0 and 1.0

4. If unsure about any transaction:
   - Set category = "OTHER"
   - Set subcategory = "UNCATEGORIZED"
   - Set confidence <= 0.5

5. Output MUST be a JSON array with one object per transaction in the SAME ORDER as input.

Transactions (JSON array):
{TRANSACTIONS_JSON}

Return ONLY a JSON array in this exact format:
[
  {{
    "category": "...",
    "subcategory": "...",
    "confidence": 0.0
  }},
  ...
]
"""

# def gemini_llm_client(prompt: str) -> str:
#     """Sends prompt to Gemini and returns raw text response."""
#     model = genai.GenerativeModel(
#         model_name="gemini-2.5-flash-lite",
#         generation_config={
#             "temperature": 0.0,
#             "max_output_tokens": 2000  # Increased for batch
#         }
#     )
    
#     response = model.generate_content(prompt).text
    
#     # Remove markdown fences if present
#     text = response.strip()
#     if text.startswith("```"):
#         text = text.strip("`")
#         if text.lower().startswith("json"):
#             text = text[4:].strip()
    
#     return text

def cohere_llm_client(prompt: str) -> str:
    """Sends prompt to Cohere Chat API and returns raw text response."""
    response = co.chat(
        model="command-a-03-2025",
        message=prompt,
        temperature=0.0,
        max_tokens=2000  # Increased for batch
    )
    
    text = response.text.strip()
    
    # Remove markdown fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    
    return text

def build_llm_payload(row):
    return {
        "counterparty_name": row["COUNTERPARTY_NAME"],
        "counterparty_vpa": row["COUNTERPARTY_VPA"],
        "remarks": row["REMARKS"],
        "txn_type": row["TXN_TYPE"],
        "direction": row["DIRECTION"],
        "amount": row["AMOUNT"],
        "date": row["DATE"] if "DATE" in row else None
    }

def batch_llm_classify(rows_df, llm_client, batch_size=30, max_retries=3):
    """
    Classify multiple transactions in batches to save API calls.
    
    Args:
        rows_df: DataFrame containing transactions to classify
        llm_client: Function to call LLM (gemini_llm_client or cohere_llm_client)
        batch_size: Number of transactions per batch (default 30)
        max_retries: Maximum retry attempts for failed batches
    
    Returns:
        Dictionary mapping row indices to classification results
    """
    results = {}
    
    # Convert DataFrame rows to list of dictionaries with indices
    rows_with_indices = []
    for idx, row in rows_df.iterrows():
        rows_with_indices.append({
            "index": idx,
            "payload": build_llm_payload(row)
        })
    
    # Process in batches
    total_batches = (len(rows_with_indices) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(rows_with_indices))
        batch_items = rows_with_indices[start_idx:end_idx]
        
        print(f"Processing batch {batch_num + 1}/{total_batches} with {len(batch_items)} transactions")
        
        # Prepare batch payload
        transactions_json = json.dumps(
            [item["payload"] for item in batch_items], 
            ensure_ascii=False,
            indent=2
        )
        
        prompt = BATCH_PROMPT_TEMPLATE.format(
            CATEGORIES=CATEGORIES,
            SUBCATEGORIES=SUBCATEGORIES,
            TRANSACTIONS_JSON=transactions_json
        )
        
        # Try classification with retries
        for attempt in range(max_retries):
            try:
                # Add delay between API calls to avoid rate limiting
                if attempt > 0:
                    delay = 2 ** attempt  # Exponential backoff
                    print(f"Retry attempt {attempt + 1}/{max_retries}, waiting {delay}s...")
                    # time.sleep(delay)
                
                response = llm_client(prompt)
                batch_results = json.loads(response)
                
                # Validate batch results
                if not isinstance(batch_results, list):
                    raise ValueError(f"Expected JSON array, got {type(batch_results)}")
                
                if len(batch_results) != len(batch_items):
                    raise ValueError(
                        f"Result count mismatch: expected {len(batch_items)}, got {len(batch_results)}"
                    )
                
                # Process each result in the batch
                for i, item in enumerate(batch_items):
                    result = batch_results[i]
                    
                    # Validate result structure
                    if not isinstance(result, dict):
                        print(f"Warning: Invalid result type at index {item['index']}: {type(result)}")
                        continue
                    
                    if "category" not in result or "subcategory" not in result:
                        print(f"Warning: Missing keys in result at index {item['index']}: {result}")
                        continue
                    
                    # Validate category and subcategory
                    if result["category"] not in CATEGORIES:
                        print(f"Warning: Invalid category at index {item['index']}: {result['category']}")
                        result["category"] = "OTHER"
                    
                    if result["subcategory"] not in SUBCATEGORIES.get(result["category"], ["UNCATEGORIZED"]):
                        result["subcategory"] = "UNCATEGORIZED"
                    
                    # Validate confidence
                    confidence = result.get("confidence", 0.0)
                    if not isinstance(confidence, (int, float)):
                        confidence = 0.0
                    elif confidence < 0.0:
                        confidence = 0.0
                    elif confidence > 1.0:
                        confidence = 1.0
                    
                    # Store result
                    results[item["index"]] = {
                        "category": result["category"],
                        "subcategory": result["subcategory"],
                        "confidence": float(confidence)
                    }
                
                print(f"✓ Batch {batch_num + 1} processed successfully")
                break  # Break retry loop on success
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error in batch {batch_num + 1}, attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    print(f"Failed to process batch {batch_num + 1} after {max_retries} attempts")
                    # Assign default classification for failed batch
                    for item in batch_items:
                        results[item["index"]] = {
                            "category": "OTHER",
                            "subcategory": "UNCATEGORIZED",
                            "confidence": 0.3
                        }
                    
            except Exception as e:
                print(f"Error in batch {batch_num + 1}, attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    print(f"Failed to process batch {batch_num + 1} after {max_retries} attempts")
                    # Assign default classification for failed batch
                    for item in batch_items:
                        results[item["index"]] = {
                            "category": "OTHER",
                            "subcategory": "UNCATEGORIZED",
                            "confidence": 0.3
                        }
        
        # Small delay between batches to avoid rate limiting
        if batch_num < total_batches - 1:
            time.sleep(1)
    
    return results

def apply_llm_results(df, llm_results):
    """Apply LLM classification results to DataFrame."""
    llm_updated = 0
    for idx, res in llm_results.items():
        if res and res["confidence"] >= LLM_THRESHOLD:
            df.at[idx, "CATEGORY"] = res["category"]
            df.at[idx, "SUBCATEGORY"] = res["subcategory"]
            df.at[idx, "CATEGORY_SOURCE"] = "LLM"
            df.at[idx, "CATEGORY_CONFIDENCE"] = res["confidence"]
            llm_updated += 1
    
    print(f"LLM categorization updated {llm_updated} transactions")
    return llm_updated

# Keep single transaction classifier for backward compatibility
def llm_classify(row, llm_client):
    """Classify a single transaction (for backward compatibility)."""
    payload = build_llm_payload(row)

    prompt = """
You are a financial transaction classifier.

You MUST follow these rules:
- Choose CATEGORY only from this list:
  {CATEGORIES}

- Choose SUBCATEGORY only from the allowed subcategories for that CATEGORY:
  {SUBCATEGORIES}

- If unsure, return:
  CATEGORY = "OTHER"
  SUBCATEGORY = "UNCATEGORIZED"
  CONFIDENCE <= 0.5

- Output ONLY valid JSON.
- No explanations outside JSON.

Transaction:
{TRANSACTION_JSON}

Return JSON in this exact format:
{{
  "category": "...",
  "subcategory": "...",
  "confidence": 0.0
}}
""".format(
    CATEGORIES=CATEGORIES,
    SUBCATEGORIES=SUBCATEGORIES,
    TRANSACTION_JSON=json.dumps(payload, ensure_ascii=False)
)

    try:
        response = llm_client(prompt)
        result = json.loads(response)
    except Exception as e:
        print(f"LLM classification error: {e}")
        return None

    # Hard validation
    if "category" not in result or "subcategory" not in result:
        return None

    if result["category"] not in CATEGORIES:
        return None

    if result["subcategory"] not in SUBCATEGORIES.get(
        result["category"], ["UNCATEGORIZED"]
    ):
        return None

    # Confidence sanity
    if not isinstance(result.get("confidence"), (int, float)):
        result["confidence"] = 0.0

    return {
        "category": result["category"],
        "subcategory": result["subcategory"],
        "confidence": float(result["confidence"])
    }