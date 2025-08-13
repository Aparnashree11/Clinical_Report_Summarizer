#!/usr/bin/env python3
"""
Clinical Text Summarization Model Training Pipeline
Fine-tunes T5 model on medical literature for clinical summarization tasks.
"""

import json
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    Trainer, TrainingArguments, DataCollatorForSeq2Seq
)

def load_medical_data_from_jsonl(file_path):
    """Load medical research data from JSONL file."""
    research_data = []
    with open(file_path, 'r') as data_file:
        for line in data_file:
            if line.strip():  # Skip empty lines
                research_data.append(json.loads(line))
    return research_data

def prepare_dataset():
    """Load and prepare training, validation, and test datasets."""
    # Load the medical research datasets
    training_records = load_medical_data_from_jsonl("/content/train_v2.json")
    validation_records = load_medical_data_from_jsonl("/content/val_v2.json")
    testing_records = load_medical_data_from_jsonl("/content/test_before_cutoff_v2.json")

    # Check first example structure
    print("Sample training record:", training_records[0])

    # Create dataset dictionary
    research_dataset = DatasetDict({
        "train": Dataset.from_list(training_records),
        "validation": Dataset.from_list(validation_records),  
        "test": Dataset.from_list(testing_records)
    })
    
    return research_dataset

def preprocess_medical_texts(example, text_tokenizer):
    """Preprocess examples for summarization model training."""
    source_text = "summarize: " + example["abstract"]
    target_summary = example["conclusion"]
    
    # Tokenize input and target texts
    tokenized_inputs = text_tokenizer(source_text, max_length=512, truncation=True, padding="max_length")
    tokenized_targets = text_tokenizer(target_summary, max_length=128, truncation=True, padding="max_length")
    
    tokenized_inputs["labels"] = tokenized_targets["input_ids"]
    return tokenized_inputs

def train_clinical_summarizer():
    """Train clinical text summarization model."""
    # Load pre-trained medical summarization model
    base_model_name = "Falconsai/medical_summarization"
    
    model_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    summarization_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    
    # Prepare datasets
    medical_dataset = prepare_dataset()
    
    # Apply preprocessing to datasets
    processed_dataset = medical_dataset.map(
        lambda x: preprocess_medical_texts(x, model_tokenizer), 
        batched=False
    )
    
    # Create smaller training subsets for faster training
    reduced_train_data = processed_dataset["train"].select(range(3000))  # first 3000 samples
    reduced_val_data = processed_dataset["validation"].select(range(300))  # first 300 samples

    # Configure training parameters
    training_config = TrainingArguments(
        output_dir="./t5-clinical-summary",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        save_strategy="epoch",
        fp16=True,
    )

    # Initialize trainer
    model_trainer = Trainer(
        model=summarization_model,
        args=training_config,
        train_dataset=reduced_train_data,
        eval_dataset=reduced_val_data,
        tokenizer=model_tokenizer,
        data_collator=DataCollatorForSeq2Seq(model_tokenizer, model=summarization_model),
    )

    # Start training process
    print("Starting model training...")
    training_results = model_trainer.train()
    print(f"Training completed. Results: {training_results}")
    
    # Evaluate the trained model
    print("Evaluating trained model...")
    evaluation_metrics = model_trainer.evaluate()
    print("Evaluation results:", evaluation_metrics)
    
    # Save the trained model and tokenizer
    output_directory = "./t5-clinical-summary"
    print(f"Saving model to {output_directory}...")
    
    model_trainer.save_model(output_directory)  # saves model weights + config
    model_tokenizer.save_pretrained(output_directory)  # saves tokenizer vocab + config
    
    print("Model and tokenizer saved successfully.")
    
    return model_trainer, model_tokenizer

def test_trained_summarizer(model_path="./t5-clinical-summary"):
    """Test the trained summarization model."""
    # Load the fine-tuned model and tokenizer
    trained_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model_tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Sample medical abstract for testing
    test_abstract = """
    Primary: Acute on chronic heart failure (NYHA Class III)
    Secondary:
    - Hypertension, well-controlled
    - Type 2 Diabetes Mellitus, suboptimally controlled
    - CKD Stage 2 (baseline creatinine)
    History of Present Illness
    Mr. John Smith, a 64-year-old male with a known history of hypertension, diabetes, and CKD,
    presented to the emergency department with increasing shortness of breath, bilateral leg swelling,
    and orthopnea over the past five days. He denied chest pain or palpitations. Physical examination
    """

    # Prepare input for the model
    summarization_input = "summarize: " + test_abstract
    model_inputs = model_tokenizer(summarization_input, return_tensors="pt", max_length=512, truncation=True)

    trained_model.eval()

    # Generate summary
    summary_output = trained_model.generate(
        **model_inputs,
        max_length=128,
        do_sample=False  # greedy decoding
    )

    # Decode and display result
    generated_summary = model_tokenizer.decode(summary_output[0], skip_special_tokens=True)
    print("Generated Summary:", generated_summary)
    print("Generated token IDs:", summary_output[0].tolist())
    print("Decoded (no filtering):", model_tokenizer.decode(summary_output[0], skip_special_tokens=False))
    
    return generated_summary

if __name__ == "__main__":
    # Install required packages (uncomment if needed)
    # import subprocess
    # subprocess.run(["pip", "install", "torch", "transformers", "datasets", "sentencepiece"])
    
    print("Starting clinical summarization model training...")
    
    # Train the model
    trainer, tokenizer = train_clinical_summarizer()
    
    print("\nTesting the trained model...")
    # Test the trained model
    test_summary = test_trained_summarizer()
    
    print("\nTraining and testing completed successfully!")