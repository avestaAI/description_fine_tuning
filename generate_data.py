from dataloader.dataloader import DataLoader
from configs.config import CFG
import time

def main():
    timeStart = time.time()
    loader = DataLoader()

    start = CFG["data"]["raw"]["start_idx"]
    end = CFG["data"]["raw"]["end_idx"]

    dataset = loader.load_raw_data(CFG["data"]["raw"]["path"], start, end)
    processed_dataset = loader.preprocess_raw_data(dataset, start)
    dataset_with_tags = loader.generate_tags(processed_dataset)

    loader.save_generated_dataset(dataset_with_tags, f"./data/interim/generated/data_{1}_to_{2}.json")
    timeEnd = time.time()

    print("Elapsed time", (timeEnd - timeStart) / 60)
    
if __name__ == "__main__":
    main()