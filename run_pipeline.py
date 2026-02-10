from src.preprocessing.preprocess import preprocess
from src.categorization.categorize import categorize

def run_pipeline():
    print("Running preprocessing...")
    preprocess()
    print("Running categorization...")
    categorize()
    print("Pipeline complete.")

if __name__ == "__main__":
    run_pipeline()