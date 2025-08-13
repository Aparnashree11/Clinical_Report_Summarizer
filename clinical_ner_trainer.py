#!/usr/bin/env python3
"""
Clinical Named Entity Recognition (NER) Model Training Pipeline
Trains ClinicalBERT for medical entity extraction from clinical texts.
"""

import gzip
import json
import numpy as np
import evaluate
import torch
from collections import defaultdict
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    Trainer, TrainingArguments, DataCollatorForTokenClassification
)

def parse_pubtator_corpus(corpus_path):
    """Parse PubTator format corpus for NER training data."""
    with gzip.open(corpus_path, 'rt', encoding='utf-8') as corpus_file:
        corpus_documents = []
        current_document = {"id": None, "text": "", "entities": []}

        for text_line in corpus_file:
            text_line = text_line.strip()
            if not text_line:
                if current_document["id"]:
                    corpus_documents.append(current_document)
                current_document = {"id": None, "text": "", "entities": []}
                continue

            if '|t|' in text_line or '|a|' in text_line:
                # Title or abstract line
                line_parts = text_line.split('|')
                document_id = line_parts[0]
                text_content = line_parts[2]
                current_document["id"] = document_id
                current_document["text"] += " " + text_content
            else:
                # Annotation line
                annotation_parts = text_line.split('\t')
                if len(annotation_parts) < 6:
                    continue
                medical_entity = {
                    "start": int(annotation_parts[1]),
                    "end": int(annotation_parts[2]),
                    "type": annotation_parts[4],
                    "text": annotation_parts[3]
                }
                current_document["entities"].append(medical_entity)

        # Add the last document
        if current_document["id"]:
            corpus_documents.append(current_document)

    return corpus_documents

def create_label_vocabulary(medical_corpus):
    """Create label vocabulary from medical entity types."""
    distinct_entity_types = set()
    for document in medical_corpus:
        for entity in document["entities"]:
            distinct_entity_types.add(entity["type"])

    # Create BIO tagging scheme
    entity_labels = ["O"]  # outside label
    for entity_type in sorted(distinct_entity_types):
        entity_labels.append(f"B-{entity_type}")
        entity_labels.append(f"I-{entity_type}")

    label_to_index = {label: idx for idx, label in enumerate(entity_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    
    return entity_labels, label_to_index, index_to_label

def convert_document_to_tokens(document, text_tokenizer, label_mapping):
    """Convert document to tokenized format with BIO labels."""
    document_text = document["text"]
    entity_annotations = document["entities"]
    entity_spans = [(e["start"], e["end"], e["type"]) for e in entity_annotations]

    text_encoding = text_tokenizer(document_text, return_offsets_mapping=True, truncation=True, max_length=180)
    token_list = text_tokenizer.convert_ids_to_tokens(text_encoding["input_ids"])
    token_offsets = text_encoding["offset_mapping"]

    # Mark special tokens offsets as None
    token_offsets = [(start, end) if (start != 0 or end != 0) else None for start, end in token_offsets]

    bio_labels = ["O"] * len(token_list)

    for token_idx, offset in enumerate(token_offsets):
        if offset is None:
            continue
        token_start, token_end = offset
        for entity_start, entity_end, entity_type in entity_spans:
            if token_start >= entity_start and token_end <= entity_end:
                label_prefix = "B" if token_start == entity_start else "I"
                bio_labels[token_idx] = f"{label_prefix}-{entity_type}"
                break

    label_indices = [label_mapping[label] for label in bio_labels]

    return {
        "input_ids": text_encoding["input_ids"],
        "attention_mask": text_encoding["attention_mask"],
        "labels": label_indices,
    }

def train_clinical_ner_model():
    """Train Clinical NER model."""
    print("Loading and parsing PubTator corpus...")
    
    # Parse the medical corpus
    medical_documents = parse_pubtator_corpus("corpus_pubtator.txt.gz")
    print(f"Loaded {len(medical_documents)} medical documents.")
    print("Sample document:", medical_documents[0])
    
    # Create label vocabulary
    print("Creating label vocabulary...")
    label_list, label_to_id, id_to_label = create_label_vocabulary(medical_documents)
    print(f"Created {len(label_list)} labels: {label_list[:10]}...")  # show first 10
    
    # Load tokenizer and model
    print("Loading ClinicalBERT model...")
    model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
    clinical_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # Convert documents to dataset
    print("Converting documents to tokenized dataset...")
    document_dataset = Dataset.from_list(medical_documents)
    
    # Map to tokenized + label dataset
    tokenized_corpus = document_dataset.map(
        lambda doc: convert_document_to_tokens(doc, clinical_tokenizer, label_to_id), 
        batched=False
    )
    
    # Remove unused columns
    tokenized_corpus = tokenized_corpus.remove_columns(["id", "text", "entities"])
    
    # Split train/test
    dataset_split = tokenized_corpus.train_test_split(test_size=0.1)
    training_data = dataset_split["train"]
    evaluation_data = dataset_split["test"]
    
    # Load model for token classification
    print("Initializing model for token classification...")
    ner_model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    
    # Prepare data collator
    token_data_collator = DataCollatorForTokenClassification(clinical_tokenizer)
    
    # Define evaluation metric
    evaluation_metric = evaluate.load("seqeval")
    
    def calculate_metrics(predictions):
        pred_logits, true_labels = predictions
        predicted_indices = np.argmax(pred_logits, axis=2)
        
        # Remove ignored index (-100) and convert to label strings
        true_label_strings = [[id_to_label[label] for label in label_seq if label != -100] for label_seq in true_labels]
        pred_label_strings = [
            [id_to_label[pred] for (pred, label) in zip(pred_seq, label_seq) if label != -100]
            for pred_seq, label_seq in zip(predicted_indices, true_labels)
        ]
        
        metric_results = evaluation_metric.compute(predictions=pred_label_strings, references=true_label_strings)
        return {
            "precision": metric_results["overall_precision"],
            "recall": metric_results["overall_recall"],
            "f1": metric_results["overall_f1"],
            "accuracy": metric_results["overall_accuracy"],
        }
    
    # Setup training configuration
    training_configuration = TrainingArguments(
        output_dir="./clinicalbert-ner",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        fp16=True,  # mixed precision
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    model_trainer = Trainer(
        model=ner_model,
        args=training_configuration,
        train_dataset=training_data,
        eval_dataset=evaluation_data,
        tokenizer=clinical_tokenizer,
        data_collator=token_data_collator,
        compute_metrics=calculate_metrics,
    )
    
    # Train the model
    print("Starting NER model training...")
    training_output = model_trainer.train()
    print(f"Training completed: {training_output}")
    
    # Final evaluation
    print("Performing final evaluation...")
    final_evaluation = model_trainer.evaluate(evaluation_data)
    print("Final evaluation metrics:", final_evaluation)
    
    # Save model and tokenizer
    output_path = "./clinicalbert-ner-final"
    print(f"Saving model to {output_path}...")
    
    model_trainer.save_model(output_path)  # saves model + tokenizer + config
    clinical_tokenizer.save_pretrained(output_path)
    
    # Save label mappings
    with open(f"{output_path}/label2id.json", "w") as label_file:
        json.dump(label_to_id, label_file)
    with open(f"{output_path}/id2label.json", "w") as id_file:
        json.dump(id_to_label, id_file)
    
    print("Model and tokenizer saved successfully.")
    return model_trainer, clinical_tokenizer, id_to_label

def test_ner_model(model_directory="./clinicalbert-ner-final"):
    """Test the trained NER model."""
    print("Loading trained NER model for testing...")
    
    # Load model and tokenizer
    test_tokenizer = AutoTokenizer.from_pretrained(model_directory)
    test_model = AutoModelForTokenClassification.from_pretrained(model_directory)
    test_model.eval()
    
    # Load label mappings
    with open(f"{model_directory}/id2label.json", "r") as id_label_file:
        id_to_label_map = json.load(id_label_file)
    id_to_label_map = {int(key): value for key, value in id_to_label_map.items()}
    
    def predict_entities(input_text):
        """Predict medical entities in text."""
        text_encoding = test_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=180, return_offsets_mapping=True)
        token_ids = text_encoding["input_ids"]
        attention_mask = text_encoding["attention_mask"]
        character_offsets = text_encoding["offset_mapping"][0]
        
        with torch.no_grad():
            model_predictions = test_model(token_ids, attention_mask=attention_mask)
            prediction_logits = model_predictions.logits
        
        predicted_labels = torch.argmax(prediction_logits, dim=2)[0].tolist()
        token_sequence = test_tokenizer.convert_ids_to_tokens(token_ids[0])
        entity_tags = [id_to_label_map[pred] for pred in predicted_labels]
        
        # Debug output
        for token, tag in zip(token_sequence, entity_tags):
            print(f"{token}: {tag}")
        
        extracted_entities = []
        current_entity = None
        
        for token_idx, (token, tag, offset) in enumerate(zip(token_sequence, entity_tags, character_offsets)):
            # Skip special tokens
            if offset[0] == 0 and offset[1] == 0:
                continue
            
            char_start = offset[0].item() if hasattr(offset[0], "item") else offset[0]
            char_end = offset[1].item() if hasattr(offset[1], "item") else offset[1]
            
            if tag == "O":
                if current_entity:
                    extracted_entities.append(current_entity)
                    current_entity = None
            else:
                tag_prefix, entity_type = tag.split("-", 1)
                
                # Handle multiple types separated by commas
                if "," in entity_type:
                    entity_type = entity_type.split(",")[0]
                
                if tag_prefix == "B":
                    if current_entity:
                        extracted_entities.append(current_entity)
                    current_entity = {
                        "type": entity_type,
                        "start": char_start,
                        "end": char_end,
                        "text": input_text[char_start:char_end],
                    }
                elif tag_prefix == "I":
                    if current_entity and current_entity["type"] == entity_type:
                        current_entity["end"] = char_end
                        current_entity["text"] = input_text[current_entity["start"]:current_entity["end"]]
                    else:
                        # Treat I- without current entity as B-
                        if current_entity:
                            extracted_entities.append(current_entity)
                        current_entity = {
                            "type": entity_type,
                            "start": char_start,
                            "end": char_end,
                            "text": input_text[char_start:char_end],
                        }
        
        if current_entity:
            extracted_entities.append(current_entity)
        
        return extracted_entities
    
    def merge_adjacent_entities(entity_list):
        """Merge adjacent entities of the same type."""
        if not entity_list:
            return []
        
        merged_list = [entity_list[0]]
        
        for current in entity_list[1:]:
            previous = merged_list[-1]
            
            # Check if types are same and spans are contiguous
            if (
                current["type"] == previous["type"] and
                previous["end"] == current["start"]
            ):
                # Merge spans
                previous["end"] = current["end"]
                previous["text"] = previous["text"] + current["text"]
            else:
                merged_list.append(current)
        
        return merged_list
    
    # Test with sample text
    sample_text = """background : hypertension, diabetes, and CKD are a common cause of acute decompensated heart failure. 
    the present report describes the history of acute decompensated heart failure. the patient presented with increasing 
    shortness of breath and bilateral leg swelling over the past five days. he denied chest pain or palpitations. 
    physical examination revealed bibasilar crackles, elevated jugular venous pressure, and 2+ pitting edema. 
    he was admitted to the emergency department for management of acute decompensated heart failure."""
    
    print("Testing NER model with sample text...")
    detected_entities = predict_entities(sample_text)
    final_entities = merge_adjacent_entities(detected_entities)
    print("Predicted entities:", [entity["text"] for entity in final_entities])
    
    return final_entities

if __name__ == "__main__":
    # Install required packages (uncomment if needed)
    # import subprocess
    # subprocess.run(["pip", "install", "seqeval"])
    
    print("Starting Clinical NER model training...")
    
    # Train the model
    trainer, tokenizer, label_mapping = train_clinical_ner_model()
    
    print("\nTesting the trained NER model...")
    # Test the trained model
    test_entities = test_ner_model()
    
    print("\nNER training and testing completed successfully!")