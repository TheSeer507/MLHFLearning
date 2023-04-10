#1. Prepare dataset
#2. load pretrained Tokenizer, call it with dataset -> encoding
#3. build PyTorch Dataset with Encodings
#4. Load pretrained model
#5. a) Load Trainer and train int
#   b) native PyTorch training loop


from transformers import Trainer, TrainingArguments 

training_args = TrainingArguments("test-trainer")

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

#This is a brief template on how to train a model using our own dataset.