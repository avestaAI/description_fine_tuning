from model.pythia import Pythia
from executors.pythiatrainer import PythiaTrainer

def main():
    pythia_model = Pythia()
    base_model = pythia_model.build()

    pythia_trainer = PythiaTrainer(base_model)
    pythia_trainer.load_data("data/final/finetuning_dataset.jsonl")
    pythia_trainer.train()
    
if __name__ == "__main__":
    main()