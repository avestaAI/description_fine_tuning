from dataloader.dataloader import DataLoader
from configs.config import CFG

def create_dataset_jsonl():
    dloader = DataLoader()
    processed_dataset = dloader.load_processed_data("./data/interim/data.pickle")
    dloader.create_final_dataset(processed_dataset)

def main():
    create_dataset_jsonl()
    
if __name__ == "__main__":
    main()