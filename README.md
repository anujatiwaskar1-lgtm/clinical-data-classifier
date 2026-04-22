Clinical Data Classifier
Overview

The Clinical Data Classifier is a machine learning-based web application designed to process unstructured clinical text and convert it into structured, interpretable data. The system performs multi-label classification to identify relevant clinical categories and applies rule-based and model-assisted extraction to retrieve key entities from the input text.

The application is implemented using Streamlit for the user interface and scikit-learn for model training and inference.

Problem Statement
Clinical data is predominantly stored in unstructured textual formats such as physician notes, laboratory reports, and discharge summaries. This lack of structure limits downstream processing, analysis, and interoperability.

This project addresses the problem by:
-Classifying clinical text into predefined categories
-Extracting structured fields from free-text input
-Providing interpretable outputs with associated confidence scores

Features
-Multi-label classification of clinical text across predefined medical categories
-Probabilistic confidence scoring for each predicted category
-Structured entity extraction (e.g., demographics, vital signs, symptoms)
-Interactive visualization of classification confidence
-Export of extracted data in CSV format
-Lightweight web interface for real-time inference

System Architecture
1.Input Layer
-User-provided clinical text via a web interface
2.Preprocessing
-Tokenization and vectorization using trained feature extractors
3.Classification
-Supervised machine learning model trained on labeled clinical transcription data
4.Post-processing
-Structured entity extraction using deterministic parsing logic
5.Output Layer
-Category predictions with confidence scores
-Extracted structured fields
-Tabular and graphical representations

Technology Stack
-Interface: Streamlit
-Machine Learning: scikit-learn
-Data Processing: pandas
-Visualization: plotly
-Model Serialization: pickle

Project Structure
clinical-data-classifier/
│
├── app.py               # Streamlit application entry point
├── classifier.py        # Model training and inference logic
├── model.pkl            # Serialized trained model
├── le.pkl               # Label encoder
├── requirements.txt     # Dependency specification
└── README.md            # Documentation

Usage
-Launch the application locally or via deployment
-Provide clinical text input (e.g., notes, reports, prescriptions)
-Execute classification
-Review predicted categories, confidence scores, and extracted entities
-Optionally export results as CSV

Output Description
1.Predicted clinical categories with associated confidence values
2.Extracted structured attributes such as:
   -Demographic information
   -Vital signs
   -Symptoms and observations

Limitations
-Performance is dependent on the quality and distribution of training data
-Entity extraction is partially rule-based and may not generalize to all input variations
-Not optimized for domain-specific edge cases or highly specialized clinical terminology

Disclaimer
This system is intended for research and educational purposes only. It is not validated for clinical use and should not be used for diagnostic or medical decision-making.

Author
Anuja Mohan Tiwaskar
