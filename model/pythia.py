from transformers import AutoModelForCausalLM
import logging
import torch

logger = logging.getLogger(__name__)
global_config = None

class Pythia:

    def __init__(self) -> None:
        self.model_name = "EleutherAI/pythia-70m"
        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def build(self):
        self._recognize_device()
        self._load_model_to_device()
        
        return self.base_model
        

    def _recognize_device(self):
        device_count = torch.cuda.device_count()
        if device_count > 0:
            logger.debug("Select GPU device")
            self.device = torch.device("cuda")
        else:
            logger.debug("Select CPU device")
            self.device = torch.device("cpu")


    def _load_model_to_device(self):
        self.base_model.to(self.device)