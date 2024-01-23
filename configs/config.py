CFG = {
    "data": {
        "raw": {
            "path": "./data/raw/data.json",
            "start_idx": 301,
            "end_idx": 400
        },
        "interim": {
            "generated": {
                "path": "./data/interim/generated/",
            },
            "corrected": {
                "path": "./data/interim/corrected/",
            }
        },
        "final": {
            "path": "./data/final/finetuning_dataset.jsonl",
        }
    },

    "training": {
        "epochs": 50,
    }
}