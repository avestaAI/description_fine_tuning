from dataloader.dataloader import DataLoader
from transformers import TrainingArguments
import torch
from utils.utilities import Trainer

class PythiaTrainer:

    def __init__(self, model) -> None:
        self.train_dataset = None
        self.test_dataset = None
        self.epochs = 10
        self.model = model
        self.trained_model_name = f"rev_desc_to_tags_{self.epochs}_steps"
        self.output_dir = "./saved_models/" + self.trained_model_name


    def load_data(self, path):
        loader = DataLoader()
        self.train_dataset, self.test_dataset = loader.load_and_tokenize_final_dataset(path)


    def _create_training_arguments(self):
        self.training_args = TrainingArguments(
            # Learning rate
            learning_rate=1.0e-5,

            # Number of training epochs
            num_train_epochs=1,

            # Max steps to train for (each step is a batch of data)
            # Overrides num_train_epochs, if not -1
            max_steps=self.epochs,

            # Batch size for training
            per_device_train_batch_size=1,

            # Directory to save model checkpoints
            output_dir=self.output_dir,

            # Other arguments
            overwrite_output_dir=False, # Overwrite the content of the output directory
            disable_tqdm=False, # Disable progress bars
            eval_steps=120, # Number of update steps between two evaluations
            save_steps=120, # After # steps model is saved
            warmup_steps=1, # Number of warmup steps for learning rate scheduler
            per_device_eval_batch_size=1, # Batch size for evaluation
            evaluation_strategy="steps",
            logging_strategy="steps",
            logging_steps=1,
            optim="adafactor",
            gradient_accumulation_steps = 4,
            gradient_checkpointing=False,

            # Parameters for early stopping
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )


    def _calculate_model_flops(self):
        self.model_flops = (
        self.model.floating_point_ops(
            {"input_ids": torch.zeros((1, 2048))}
        )
        * self.training_args.gradient_accumulation_steps
        )

        print(self.model)
        print("Memory footprint", self.model.get_memory_footprint() / 1e9, "GB")
        print("Flops", self.model_flops / 1e9, "GFLOPs")


    def _save_model(self):
        save_dir = f'{self.output_dir}/'
        self.trainer.save_model(save_dir)
        print("Saved model to:", save_dir)


    def train(self):
        self._create_training_arguments()
        self._calculate_model_flops()

        self.trainer = Trainer(
            model=self.model,
            model_flops=self.model_flops,
            total_steps=self.epochs,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            args=self.training_args
        )

        training_output = self.trainer.train()
        self._save_model()

        return training_output

