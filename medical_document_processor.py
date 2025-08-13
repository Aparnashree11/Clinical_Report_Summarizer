import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForTokenClassification
import json
import re
from typing import List, Set

def extract_content_from_pdf(pdf_file_path):
    """Extract text content from PDF file."""
    extracted_content = ""
    with pdfplumber.open(pdf_file_path) as document:
        for page_obj in document.pages:
            extracted_content += page_obj.extract_text()
    return extracted_content

# Load fine-tuned clinical summarization model
clinical_summarizer = AutoModelForSeq2SeqLM.from_pretrained("content/t5-clinical-summary")
summary_tokenizer = AutoTokenizer.from_pretrained("content/t5-clinical-summary")

def generate_clinical_summary(document_content):
    """Generate clinical summary from document content."""
    input_prompt = "summarize: " + document_content
    tokenized_input = summary_tokenizer(input_prompt, return_tensors="pt", max_length=512, truncation=True)
    
    clinical_summarizer.eval()
    
    # Generate summary using the model
    generated_output = clinical_summarizer.generate(
        **tokenized_input,
        max_length=128,
        do_sample=False  # greedy decoding
    )
    
    clinical_summary = summary_tokenizer.decode(generated_output[0], skip_special_tokens=True)
    return clinical_summary

# Load NER model for entity extraction
ner_model_directory = "./test/clinicalbert-ner-final"
entity_tokenizer = AutoTokenizer.from_pretrained(ner_model_directory)
entity_model = AutoModelForTokenClassification.from_pretrained(ner_model_directory)
entity_model.eval()

# Load label mappings
with open(f"{ner_model_directory}/id2label.json", "r") as label_file:
    index_to_label = json.load(label_file)
index_to_label = {int(key): value for key, value in index_to_label.items()}

def extract_medical_entities(input_text):
    """Extract medical entities from text using NER model."""
    text_encoding = entity_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=180, return_offsets_mapping=True)
    token_ids = text_encoding["input_ids"]
    attention_weights = text_encoding["attention_mask"]
    character_offsets = text_encoding["offset_mapping"][0]

    with torch.no_grad():
        model_output = entity_model(token_ids, attention_mask=attention_weights)
        prediction_logits = model_output.logits

    predicted_labels = torch.argmax(prediction_logits, dim=2)[0].tolist()
    token_list = entity_tokenizer.convert_ids_to_tokens(token_ids[0])
    entity_tags = [index_to_label[prediction] for prediction in predicted_labels]

    identified_entities = []
    active_entity = None

    for token_idx, (token, tag, offset) in enumerate(zip(token_list, entity_tags, character_offsets)):
        # Skip special tokens (offsets == (0, 0))
        if offset[0] == 0 and offset[1] == 0:
            continue

        # Convert tensor to int if needed
        char_start = offset[0].item() if hasattr(offset[0], "item") else offset[0]
        char_end = offset[1].item() if hasattr(offset[1], "item") else offset[1]

        if tag == "O":
            if active_entity:
                identified_entities.append(active_entity)
                active_entity = None
        else:
            tag_prefix, entity_category = tag.split("-", 1)

            # Handle multiple types separated by commas
            if "," in entity_category:
                entity_category = entity_category.split(",")[0]  # take the first type

            if tag_prefix == "B":
                if active_entity:
                    identified_entities.append(active_entity)
                active_entity = {
                    "type": entity_category,
                    "start": char_start,
                    "end": char_end,
                    "text": input_text[char_start:char_end],
                }
            elif tag_prefix == "I":
                if active_entity and active_entity["type"] == entity_category:
                    active_entity["end"] = char_end
                    active_entity["text"] = input_text[active_entity["start"]:active_entity["end"]]
                else:
                    # Treat I- without current entity as B-
                    if active_entity:
                        identified_entities.append(active_entity)
                    active_entity = {
                        "type": entity_category,
                        "start": char_start,
                        "end": char_end,
                        "text": input_text[char_start:char_end],
                    }

    if active_entity:
        identified_entities.append(active_entity)

    return identified_entities

def consolidate_adjacent_entities(entity_list):
    """Merge adjacent entities of the same type."""
    if not entity_list:
        return []

    consolidated_entities = [entity_list[0]]

    for current_entity in entity_list[1:]:
        previous_entity = consolidated_entities[-1]

        # Check if types are same and spans are contiguous
        if (
            current_entity["type"] == previous_entity["type"] and
            previous_entity["end"] == current_entity["start"]
        ):
            # Merge spans
            previous_entity["end"] = current_entity["end"]
            previous_entity["text"] = previous_entity["text"] + current_entity["text"]
        else:
            consolidated_entities.append(current_entity)

    return consolidated_entities

def normalize_medical_terminology(entity_texts: List[str]) -> List[str]:
    """Clean and normalize medical entity terms."""
    # Convert to lowercase and strip whitespace
    normalized_terms = [term.lower().strip() for term in entity_texts if term.strip()]

    # Medical abbreviation and term corrections
    medical_corrections = {
        # Cardiovascular terms
        r'\bhtn\b': 'hypertension',
        r'\bhypertens\b': 'hypertension',
        r'\bmi\b': 'myocardial infarction',
        r'\baf\b': 'atrial fibrillation',
        r'\bchf\b': 'congestive heart failure',
        r'\badhf\b': 'acute decompensated heart failure',
        r'\bdecompensated heart\b': 'decompensated heart failure',
        r'\bacute decompensated heart\b': 'acute decompensated heart failure',
        r'\bjvp\b': 'jugular venous pressure',
        r'\bjugular venous\b': 'jugular venous distension',

        # Respiratory terms
        r'\bsob\b': 'shortness of breath',
        r'\bcrack\b': 'crackles',
        r'\bpe\b': 'pulmonary embolism',
        r'\bosa\b': 'obstructive sleep apnea',

        # Neurological terms
        r'\bcva\b': 'cerebrovascular accident',
        r'\btia\b': 'transient ischemic attack',

        # Lab and diagnostic terms
        r'\bck\b': 'creatine kinase',
        r'\bldh\b': 'lactate dehydrogenase',
        r'\bbnp\b': 'b-type natriuretic peptide',
        r'\bcrp\b': 'c-reactive protein',
        r'\bmri\b': 'magnetic resonance imaging',
        r'\bct\b': 'computed tomography',

        # Symptom terms
        r'\bpal\b': 'palpitations',
        r'\bpit\b': 'pitting edema',
        r'\bpitting ed\b': 'pitting edema',
        r'\bod\b': 'overdose',

        # Remove non-medical noise terms
        r'\bila\b': '',
        r'\bbi\b': '',
        r'\bbas\b': '',
        r'\bphysical\b': '',
        r'\bemergency\b': '',
        r'\bleg\b': '',
        r'\bchest\b': '',
        r'\bshortness of\b': ''
    }

    # Process multi-word combinations
    combined_medical_terms = set()
    for term_idx in range(len(normalized_terms)):
        for n_gram in [3, 2]:  # try 3-grams first, then 2-grams
            phrase = " ".join(normalized_terms[term_idx:term_idx + n_gram]).strip()
            for pattern, replacement in medical_corrections.items():
                if re.search(pattern, phrase, flags=re.IGNORECASE):
                    if replacement:  # skip empty replacements
                        combined_medical_terms.add(replacement.lower())
    
    # Process individual terms
    processed_terms: Set[str] = set()
    for term in normalized_terms:
        corrected_term = medical_corrections.get(term, term)
        if corrected_term and len(corrected_term) > 2:
            processed_terms.add(corrected_term)

    # Merge and deduplicate
    final_terminology = sorted(set(processed_terms).union(combined_medical_terms))
    return final_terminology

# Load language model for term definitions
definition_model_name = "bigscience/bloom-560m"
definition_tokenizer = AutoTokenizer.from_pretrained(definition_model_name)
definition_model = AutoModelForCausalLM.from_pretrained(definition_model_name)
definition_model.eval()

def generate_term_definition(medical_term):
    """Generate definition for medical term using language model."""
    definition_prompt = f'Explain the medical term "{medical_term}" in simple words'
    model_inputs = definition_tokenizer(definition_prompt, return_tensors="pt")
    with torch.no_grad():
        generated_tokens = definition_model.generate(
            model_inputs.input_ids,
            max_new_tokens=50
        )
    return definition_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

def process_medical_document(pdf_path):
    """Complete medical document processing pipeline."""
    print("Extracting text from PDF...")
    document_text = extract_content_from_pdf(pdf_path)
    print(f"Extracted text preview: {document_text[:500]}...")
    
    print("\nGenerating clinical summary...")
    summary_text = generate_clinical_summary(document_text)
    print(f"Generated Summary: {summary_text}")
    
    print("\nExtracting medical entities...")
    raw_entities = extract_medical_entities(summary_text)
    processed_entities = consolidate_adjacent_entities(raw_entities)
    entity_text_list = [entity["text"] for entity in processed_entities]
    print(f"Identified entities: {entity_text_list}")
    
    print("\nNormalizing medical terminology...")
    normalized_medical_terms = normalize_medical_terminology(entity_text_list)
    print(f"Normalized terms: {normalized_medical_terms}")
    
    print("\nGenerating definitions for medical terms...")
    for medical_term in normalized_medical_terms:
        definition_result = generate_term_definition(medical_term)
        print(f"{medical_term}: {definition_result}")

if __name__ == "__main__":
    # Example usage
    sample_pdf_path = "Patient_Discharge_Summary_Report.pdf"
    process_medical_document(sample_pdf_path)