import json
from data import desc_dataset

if __name__ == "__main__":

    json_data = json.dumps(desc_dataset, indent = 2)
    json_file_path = "../data/interim/data.json"

    with open(json_file_path, "w") as json_file:
        json_file.write(json_data)

    print("Json data written successfully!")
