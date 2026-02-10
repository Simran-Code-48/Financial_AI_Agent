import sys
import os
from src.preprocessing.preprocess import preprocess
from src.categorization.categorize import categorize
from src.agent.run_agent import run_agent
from run_pipeline import run_pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("Banking Analyzer")
    print("=" * 30)
    print("1. Run preprocessing pipeline")
    print("2. Run categorization")
    print("3. Start agent for querying")
    print("4. Run complete pipeline")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == "1":
        preprocess()
    elif choice == "2":
        categorize()
    elif choice == "3":
        run_agent()
    elif choice == "4":
        run_pipeline()
    elif choice == "5":
        print("Exiting...")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()