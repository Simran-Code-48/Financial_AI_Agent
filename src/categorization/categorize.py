import pandas as pd
from .rule_based import rule_based_category
from .llm_classifier import batch_llm_classify, cohere_llm_client, apply_llm_results
from src.config import ( 
    PREPROCESSED_PATH, 
    RULE_CATEGORIZED_PATH,
    FINAL_CATEGORIZED_PATH, 
    LLM_CATEGORIZED_PATH,
    LLM_THRESHOLD
    )
 
def categorize():
    improved_df = pd.read_csv(PREPROCESSED_PATH)
    print(f"Loaded {len(improved_df)} transactions from {PREPROCESSED_PATH}")

    # Rule-based categorization
    print("Applying rule-based categorization...")
    category_cols = improved_df.apply(
        lambda r: pd.Series(
            rule_based_category(r),
            index=["CATEGORY", "SUBCATEGORY", "CATEGORY_SOURCE", "CATEGORY_CONFIDENCE"]
        ),
        axis=1
    )
    improved_df = pd.concat([improved_df, category_cols], axis=1)

    # Save rule-based categorization
    improved_df.to_csv(RULE_CATEGORIZED_PATH, index=False)
    print(f"Rule-based categorization saved to: {RULE_CATEGORIZED_PATH}")

    # LLM categorization for low confidence rows
    llm_candidates = improved_df[
        (improved_df["CATEGORY_SOURCE"] == "RULE") &
        (improved_df["CATEGORY_CONFIDENCE"] < LLM_THRESHOLD)
    ].copy()
    print(f"Found {len(llm_candidates)} transactions for LLM categorization (confidence < {LLM_THRESHOLD})")

    
    llm_results = {}
    if len(llm_candidates) > 0:
        print("Starting LLM batch categorization...")
        llm_results = batch_llm_classify(llm_candidates, cohere_llm_client, batch_size=30)
    # Apply LLM results
        updated_count = apply_llm_results(improved_df, llm_results)
        print(f"LLM updated {updated_count} transactions with high confidence")
    # Save LLM-enriched data
        llm_enriched = improved_df[improved_df["CATEGORY_SOURCE"] == "LLM"]
        if len(llm_enriched) > 0:
            llm_enriched.to_csv(LLM_CATEGORIZED_PATH, index=False)
            print(f"LLM-enriched transactions saved to {LLM_CATEGORIZED_PATH}")
    else:
        print("No transactions require LLM categorization")
    # Save final categorized data
    improved_df.to_csv(FINAL_CATEGORIZED_PATH, index=False)
    print(f"Final categorized data saved to {FINAL_CATEGORIZED_PATH}")
    
    # Print categorization summary
    print("\nCategorization Summary:")
    print("=" * 40)
    print(f"Total transactions: {len(improved_df)}")
    print(f"Rule-based categorizations: {len(improved_df[improved_df['CATEGORY_SOURCE'] == 'RULE'])}")
    print(f"LLM categorizations: {len(improved_df[improved_df['CATEGORY_SOURCE'] == 'LLM'])}")
    
    # Category distribution
    print("\nCategory Distribution:")
    category_counts = improved_df['CATEGORY'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(improved_df)) * 100
        print(f"  {category}: {count} transactions ({percentage:.1f}%)")
    
    return improved_df



if __name__ == "__main__":
    categorize()