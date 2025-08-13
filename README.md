# Medical Document Analyzer

A comprehensive AI-powered system for processing medical documents, generating clinical summaries, extracting medical entities, and providing plain-language explanations of medical terminology.

---

## Features

- **PDF Text Extraction**: Extract text content from medical PDF reports
- **Clinical Summarization**: Generate concise summaries using fine-tuned T5 models
- **Medical Entity Recognition**: Identify and extract medical terms using ClinicalBERT
- **Term Normalization**: Clean and standardize medical terminology
- **Interactive Definitions**: Get plain-language explanations of medical terms
- **Web Interface**: User-friendly Streamlit application

---

## Architecture

The system consists of four main components:

1. **Medical Document Processor** (`medical_document_processor.py`)
2. **Clinical Summarization Trainer** (`clinical_summarization_trainer.py`)
3. **Clinical NER Trainer** (`clinical_ner_trainer.py`)
4. **Streamlit Web Application** (`app.py`)

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Aparnashree11/Clinical_Report_Summarizer.git
```

2. **Create virtual environment**
```bash
python -m venv medical_env
source medical_env/bin/activate  # On Windows: medical_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required models**
```bash
# For summarization training
# Download MedMentions dataset

# For NER training
# Download PubTator corpus (corpus_pubtator.txt.gz)
```

---

## Models Used

### Pre-trained Models
- **Summarization**: `Falconsai/medical_summarization` (T5-based)
- **Entity Recognition**: `emilyalsentzer/Bio_ClinicalBERT`
- **Term Definition**: `bigscience/bloom-560m`

### Custom Trained Models
- **Clinical Summarizer**: Fine-tuned T5 on medical literature (MedMentions)
- **Medical NER**: ClinicalBERT fine-tuned on PubTator corpus

---

## Usage

### Web Application

1. **Start the Streamlit app**
```bash
streamlit run streamlit_medical_analyzer.py
```

2. **Upload a medical PDF document**

3. **View generated summary and medical terms**

4. **Click on medical terms for definitions**

### Command Line Usage

```python
from medical_document_processor import process_medical_document

# Process a medical PDF
pdf_path = "sample_medical_report.pdf"
process_medical_document(pdf_path)
```

---

## Training Custom Models

### 1. Clinical Summarization Model

```bash
python clinical_summarization_trainer.py
```

**Training Process:**
- Uses medical research abstracts and conclusions
- Fine-tunes T5 model for domain-specific summarization
- Evaluates on validation set
- Saves trained model to `./t5-clinical-summary/`

### 2. Medical NER Model

```bash
python clinical_ner_trainer.py
```

**Training Process:**
- Uses PubTator corpus for medical entity annotation
- Fine-tunes ClinicalBERT for token classification
- Creates BIO tagging scheme for medical entities
- Saves trained model to `./clinicalbert-ner-final/`

---

## Key Functions

### Medical Document Processor
- `extract_content_from_pdf()`: PDF text extraction
- `generate_clinical_summary()`: Text summarization
- `extract_medical_entities()`: NER entity extraction
- `normalize_medical_terminology()`: Term cleaning
- `generate_term_definition()`: Medical term explanations

### Training Scripts
- `train_clinical_summarizer()`: Train T5 summarization model
- `train_clinical_ner_model()`: Train ClinicalBERT NER model
- `test_trained_summarizer()`: Evaluate summarization performance
- `test_ner_model()`: Evaluate NER performance

---

## Configuration

### Model Parameters
```python
# Summarization
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128
LEARNING_RATE = 2e-5
BATCH_SIZE = 8

# NER
MAX_SEQUENCE_LENGTH = 180
NER_LEARNING_RATE = 2e-5
NER_BATCH_SIZE = 4
```

### Medical Term Corrections
The system includes built-in medical abbreviation expansions:
- HTN → Hypertension
- CK → Creatine Kinase
- SOB → Shortness of Breath
- JVP → Jugular Venous Pressure

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in training arguments
   - Use gradient accumulation steps

2. **Model Loading Errors**
   - Ensure model paths are correct
   - Check if models are properly saved

3. **PDF Processing Issues**
   - Verify PDF is not password protected
   - Check if PDF contains extractable text

---
     
