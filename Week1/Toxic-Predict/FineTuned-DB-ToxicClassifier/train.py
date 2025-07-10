from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import os

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set")

dataset = load_dataset(
    'csv',
    data_files={
        'train': 'C:\\Users\\NightPrince\\OneDrive\\Desktop\\Cellula-Internship\\Week1\\Toxic-Predict\\data\\train.csv',
        'validation': 'C:\\Users\\NightPrince\\OneDrive\\Desktop\\Cellula-Internship\\Week1\\Toxic-Predict\\data\\eval.csv'
    }
)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["query"],
        examples["image descriptions"],
        truncation=True,
        padding="max_length",
        max_length=150,
    )

dataset = dataset.map(tokenize_function, batched=True)

dataset = dataset.rename_column("Toxic Category Encoded", "labels")
dataset = dataset.remove_columns(['query', 'image descriptions', 'Toxic Category'])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=9)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_lin", "v_lin"],
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(model, lora_config)

id2label = {
    0: "Child Sexual Exploitation",
    1: "Elections",
    2: "Non-Violent Crimes",
    3: "Safe",
    4: "Sex-Related Crimes",
    5: "Suicide & Self-Harm",
    6: "Unknown S-Type",
    7: "Violent Crimes",
    8: "unsafe"
}
model.config.id2label = id2label
model.config.label2id = {v: k for k, v in id2label.items()}

training_args = TrainingArguments(
    output_dir="results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",              
    num_train_epochs=3,
    push_to_hub=True,
    hub_strategy="checkpoint",
    hub_model_id="NightPrince/peft-distilbert-toxic-classifier",
    hub_token=hf_token
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
)

trainer.train()


metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


trainer.push_to_hub()
