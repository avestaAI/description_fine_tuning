import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.templates import tags, template_1
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import jsonlines
import pandas as pd
import pickle

class DataLoader:
    """Data Loader Class"""

    def __init__(self) -> None:
        pass


    @staticmethod
    def load_raw_data(path, start, end):
        """Loads the data from the data.json"""

        with open(path, "r") as json_file:
            data = json.load(json_file)

        return data[start-1:end]
    

    @staticmethod
    def preprocess_raw_data(dataset, start):
        """Removes unwanted fields from the dataset (eg: _index, _type etc)"""

        processed_dataset = []
        for index, datapoint in enumerate(dataset):
            description = datapoint["_source"]["description"]
            new_datapoint = {
                "id": (start + index),
                "description": description
            }
            processed_dataset.append(new_datapoint)

        return processed_dataset
    
    
    @staticmethod
    def generate_tags(dataset):
        """Extracts tags from the descriptions using llm"""
        _ = os.getenv("OPENAI_API_KEY")
        gpt = ChatOpenAI(temperature = 0.0)

        dataset_with_tags = []

        for datapoint in dataset:
            prompt_template = ChatPromptTemplate.from_template(template_1)
            message = prompt_template.format_messages(
                description = datapoint["description"],
                tags = tags
            )
            try:
                llm_response = gpt(message)
                predicted_tags = eval(llm_response.content)
            except Exception as e:
                print("An Error occurred while get llm response:", e)

            new_datapoint = {
                "id": datapoint["id"],
                "description": datapoint["description"],
                "tags": predicted_tags
            }

            dataset_with_tags.append(new_datapoint)
            
        return dataset_with_tags
        

    @staticmethod
    def save_generated_dataset(dataset, path):
        """Saves the generated dataset with the tags to a json file"""

        try:
            with open(path, "w") as file:
                json.dump(dataset, file,indent=2)
        except Exception as e:
            print("An error occurred while saving into JSON file!", e)

        print("Data stored in json file successfully! at ",path)


    @staticmethod
    def save_corrected_dataset(path):
        """Takes the corrected json files and creates a final json lines dataset"""

        final_corrected_dataset = []

        for filename in os.listdir(path):

            with open(os.path.join(path, filename), "r") as file:
                interim_dataset = json.load(file)
                key_updated = []
                for datapoint in interim_dataset:
                    new_dict = {
                        "id": datapoint["id"],
                        "input": datapoint["description"],
                        "output": datapoint["tags"]
                    }
                    key_updated.append(new_dict)

            final_corrected_dataset.extend(key_updated)

        with jsonlines.open("dataset.jsonl", "w") as writer:
            writer.write_all(final_corrected_dataset)

            
    @staticmethod
    def tokenize_datapoint(finetuning_dataset_dict):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
        description = finetuning_dataset_dict["input"][0]
        output = finetuning_dataset_dict["output"][0]
        text = description + output

        tokenizer.pad_token = tokenizer.eos_token
        tokenized_inputs = tokenizer(
            text = text,
            return_tensors = "np",
            padding = True
        )

        max_length = min(tokenized_inputs["input_ids"].shape[1], 2048)

        tokenized_inputs = tokenizer(
            text = text,
            return_tensors = "np",
            truncation = True,
            max_length = max_length
        )
        return tokenized_inputs
    
    
    @staticmethod
    def load_and_tokenize_final_dataset(path):
        finetuning_dataset_hf = load_dataset(
            "json",
            data_files=path,
            split="train"
        )

        tokenized_dataset = finetuning_dataset_hf.map(
            DataLoader.tokenize_datapoint,
            batched=True,
            batch_size=1,
            drop_last_batch=True
        )
        tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])
        splitted_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True, seed=123)

        return splitted_dataset["train"], splitted_dataset["test"]