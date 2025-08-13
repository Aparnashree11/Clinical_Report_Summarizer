import streamlit as st
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoModelForCausalLM
import torch
import json
from typing import List, Set
import re

# ------------------ PDF CONTENT EXTRACTION ------------------ #
def extract_content_from_pdf(pdf_document):
    """Extract text content from uploaded PDF file."""
    extracted_content = ""
    with pdfplumber.open(pdf_document) as document:
        for page_obj in document.pages:
            extracted_content += page_obj.extract_text()
    return extracted_content


# ------------------ CLINICAL SUMMARIZATION ------------------ #
@st.cache_resource
def load_clinical_summarization_model():
    """Load the clinical summarization model and tokenizer."""
    clinical_summarizer = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/medical_summarization")
    summary_tokenizer = AutoTokenizer.from_pretrained("Falconsai/medical_summarization")
    clinical_summarizer.eval()
    return clinical_summarizer, summary_tokenizer

def generate_clinical_summary(document_content):
    """Generate clinical summary from document content."""
    clinical_summarizer, summary_tokenizer = load_clinical_summarization_model()
    input_prompt = "summarize: " + document_content
    tokenized_input = summary_tokenizer(input_prompt, return_tensors="pt", max_length=512, truncation=True)
    generated_output = clinical_summarizer.generate(**tokenized_input, max_length=128, do_sample=False)
    return summary_tokenizer.decode(generated_output[0], skip_special_tokens=True)


# ------------------ MEDICAL ENTITY EXTRACTION ------------------ #
@st.cache_resource
def load_entity_extraction_model():
    """Load the NER model for medical entity extraction."""
    ner_model_directory = "./test/clinicalbert-ner-final"
    entity_tokenizer = AutoTokenizer.from_pretrained(ner_model_directory)
    entity_model = AutoModelForTokenClassification.from_pretrained(ner_model_directory)
    entity_model.eval()
    with open(f"{ner_model_directory}/id2label.json", "r") as label_file:
        index_to_label = json.load(label_file)
    index_to_label = {int(key): value for key, value in index_to_label.items()}
    return entity_model, entity_tokenizer, index_to_label

def extract_medical_entities(input_text):
    """Extract medical entities from text using NER model."""
    entity_model, entity_tokenizer, index_to_label = load_entity_extraction_model()
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
        if offset[0] == 0 and offset[1] == 0:
            continue
        char_start = offset[0].item() if hasattr(offset[0], "item") else offset[0]
        char_end = offset[1].item() if hasattr(offset[1], "item") else offset[1]
        if tag == "O":
            if active_entity:
                identified_entities.append(active_entity)
                active_entity = None
        else:
            tag_prefix, entity_category = tag.split("-", 1)
            if "," in entity_category:
                entity_category = entity_category.split(",")[0]
            if tag_prefix == "B":
                if active_entity:
                    identified_entities.append(active_entity)
                active_entity = {"type": entity_category, "start": char_start, "end": char_end, "text": input_text[char_start:char_end]}
            elif tag_prefix == "I":
                if active_entity and active_entity["type"] == entity_category:
                    active_entity["end"] = char_end
                    active_entity["text"] = input_text[active_entity["start"]:active_entity["end"]]
                else:
                    if active_entity:
                        identified_entities.append(active_entity)
                    active_entity = {"type": entity_category, "start": char_start, "end": char_end, "text": input_text[char_start:char_end]}
    if active_entity:
        identified_entities.append(active_entity)
    return consolidate_adjacent_entities(identified_entities)

def consolidate_adjacent_entities(entity_list):
    """Merge adjacent entities of the same type."""
    if not entity_list:
        return []
    consolidated_entities = [entity_list[0]]
    for current_entity in entity_list[1:]:
        previous_entity = consolidated_entities[-1]
        if current_entity["type"] == previous_entity["type"] and previous_entity["end"] == current_entity["start"]:
            previous_entity["end"] = current_entity["end"]
            previous_entity["text"] += current_entity["text"]
        else:
            consolidated_entities.append(current_entity)
    return consolidated_entities


# ------------------ MEDICAL TERMINOLOGY NORMALIZATION ------------------ #
def normalize_medical_terminology(entity_texts: List[str]) -> List[str]:
    """Clean and normalize medical entity terms."""
    normalized_terms = [term.lower().strip() for term in entity_texts if term.strip()]
    medical_corrections = {
        'hypertens': 'hypertension',
        'ck': 'creatine kinase',
        'pal': 'palpitations',
        'pit': 'pitting edema',
        'pitting ed': 'pitting edema',
        'crack': '',
        'sob': '',
        'jvp': 'jugular venous pressure',
        'ila': '', 'bi': '', 'bas': '',  # noise terms
        'physical': '', 'emergency': '', 'leg': '', 'chest': '',
        'decompensated heart': 'decompensated heart failure',
        'acute decompensated heart': 'acute decompensated heart failure',
        'shortness of': '',
        'jugular venous': 'jugular venous distension'
    }
    combined_medical_terms = set()
    for term_idx in range(len(normalized_terms)):
        for n_gram in [3, 2]:
            phrase = " ".join(normalized_terms[term_idx:term_idx + n_gram])
            if phrase in medical_corrections and medical_corrections[phrase]:
                combined_medical_terms.add(medical_corrections[phrase])
    processed_terms: Set[str] = set()
    for term in normalized_terms:
        corrected_term = medical_corrections.get(term, term)
        if corrected_term and len(corrected_term) > 2:
            processed_terms.add(corrected_term)
    return sorted(set(processed_terms).union(combined_medical_terms))

def format_sentence_capitalization(text_content):
    """Capitalize sentences properly."""
    # Split on sentence endings (., ?, !)
    sentence_parts = re.split(r'([.?!]\s*)', text_content)
    # Capitalize first letter of each sentence part
    sentence_parts = [part.capitalize() for part in sentence_parts]
    return ''.join(sentence_parts)


# ------------------ MEDICAL TERM DEFINITIONS ------------------ #
@st.cache_resource
def load_definition_generator():
    """Load the model for generating medical term definitions."""
    definition_model_name = "bigscience/bloom-560m"
    definition_tokenizer = AutoTokenizer.from_pretrained(definition_model_name)
    definition_model = AutoModelForCausalLM.from_pretrained(definition_model_name)
    definition_model.eval()
    return definition_model, definition_tokenizer

def generate_term_definition(medical_term):
    """Generate definition for medical term using language model."""
    definition_model, definition_tokenizer = load_definition_generator()
    definition_prompt = f'Explain the medical term "{medical_term}" in simple words'
    model_inputs = definition_tokenizer(definition_prompt, return_tensors="pt")
    with torch.no_grad():
        generated_tokens = definition_model.generate(
            model_inputs.input_ids,
            max_new_tokens=50
        )
    response_text = definition_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return re.findall(r'\.(.*?\.)', response_text)[0].strip()


# ------------------ STREAMLIT USER INTERFACE ------------------ #
st.set_page_config(layout="wide", page_title="Medical Document Analyzer")
st.title("Medical Document Analyzer & Term Explainer")

# File upload widget
uploaded_pdf_file = st.file_uploader("Upload a medical PDF report", type="pdf")

if uploaded_pdf_file is not None:

    # Step 1: Store extracted content, summary, entities, and terms in session state
    if "document_content" not in st.session_state:
        with st.spinner("Extracting content from PDF..."):
            st.session_state.document_content = extract_content_from_pdf(uploaded_pdf_file)

    if "clinical_summary" not in st.session_state:
        with st.spinner("Generating clinical summary..."):
            st.session_state.clinical_summary = generate_clinical_summary(st.session_state.document_content)

    if "medical_entities" not in st.session_state or "normalized_medical_terms" not in st.session_state:
        with st.spinner("Identifying medical terminology..."):
            medical_entities = extract_medical_entities(st.session_state.clinical_summary)
            raw_entity_texts = [entity["text"] for entity in medical_entities]
            st.session_state.medical_entities = medical_entities
            st.session_state.normalized_medical_terms = normalize_medical_terminology(raw_entity_texts)

    # Step 2: Display clinical summary
    st.subheader("Clinical Summary")
    st.write(format_sentence_capitalization(st.session_state.clinical_summary))

    # Step 3: Display medical term buttons and lazy-load definitions
    st.markdown("---")
    st.subheader("Medical Term Definitions")

    if "medical_term_definitions" not in st.session_state:
        st.session_state.medical_term_definitions = {}

    for medical_term in st.session_state.normalized_medical_terms:
        if st.button(medical_term):
            if medical_term not in st.session_state.medical_term_definitions:
                with st.spinner(f"Generating definition for '{medical_term}'..."):
                    st.session_state.medical_term_definitions[medical_term] = generate_term_definition(medical_term)
            st.markdown(f"**{medical_term.capitalize()}**")
            st.info(st.session_state.medical_term_definitions[medical_term])