"""
classifier.py
All ML logic: preprocessing, training, multilabel prediction, entity extraction.
"""

import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.utils import resample

# ─────────────────────────────────────────────────────────
#  CATEGORY MAP  (40 Kaggle specialties → 6 categories)
# ─────────────────────────────────────────────────────────
CATEGORY_MAP = {
    "General Medicine"              : "Demographics",
    "Pediatrics - Neonatal"         : "Demographics",
    "Dentistry"                     : "Demographics",
    "Chiropractic"                  : "Demographics",
    "Hospice - Palliative Care"     : "Demographics",
    "IME-QME-Work Comp etc."        : "Demographics",
    "Emergency Room Reports"        : "Vital Signs",
    "Cardiovascular / Pulmonary"    : "Vital Signs",
    "Pulmonology"                   : "Vital Signs",
    "Sleep Medicine"                : "Vital Signs",
    "Lab Medicine - Pathology"      : "Laboratory Reports",
    "Nephrology"                    : "Laboratory Reports",
    "Endocrinology"                 : "Laboratory Reports",
    "Hematology - Oncology"         : "Laboratory Reports",
    "Rheumatology"                  : "Laboratory Reports",
    "Allergy / Immunology"          : "Laboratory Reports",
    "Pathology"                     : "Pathology Reports",
    "Autopsy"                       : "Pathology Reports",
    "Neurosurgery"                  : "Pathology Reports",
    "Surgery"                       : "Pathology Reports",
    "Orthopedic"                    : "Pathology Reports",
    "Urology"                       : "Pathology Reports",
    "Obstetrics / Gynecology"       : "Pathology Reports",
    "Gastroenterology"              : "Pathology Reports",
    "Ophthalmology"                 : "Pathology Reports",
    "Plastic Surgery"               : "Pathology Reports",
    "ENT - Otolaryngology"          : "Pathology Reports",
    "Dermatology"                   : "Pathology Reports",
    "Cosmetic / Plastic Surgery"    : "Pathology Reports",
    "Bariatrics"                    : "Pathology Reports",
    "Infectious Disease"            : "Microbiology Data",
    "Consult - History and Phy."    : "Clinical Notes",
    "Discharge Summary"             : "Clinical Notes",
    "SOAP / Chart / Progress Notes" : "Clinical Notes",
    "Office Notes"                  : "Clinical Notes",
    "Letters"                       : "Clinical Notes",
    "Physical Medicine - Rehab"     : "Clinical Notes",
    "Psychiatry / Psychology"       : "Clinical Notes",
    "Pain Management"               : "Clinical Notes",
    "Neurology"                     : "Clinical Notes",
    "Radiology"                     : "Clinical Notes",
    "Diets and Nutritions"          : "Clinical Notes",
    "Speech - Language"             : "Clinical Notes",
    "Podiatry"                      : "Clinical Notes",
}

# ─────────────────────────────────────────────────────────
#  CLINICAL ABBREVIATIONS
# ─────────────────────────────────────────────────────────
CLINICAL_ABBR = {
    "BP"   : "blood pressure",
    "HR"   : "heart rate",
    "SpO2" : "oxygen saturation",
    "WBC"  : "white blood cell",
    "RBC"  : "red blood cell",
    "CBC"  : "complete blood count",
    "HbA1c": "glycated hemoglobin",
    "CRP"  : "c reactive protein",
    "ESR"  : "erythrocyte sedimentation rate",
    "AFB"  : "acid fast bacilli",
    "MRSA" : "methicillin resistant staphylococcus",
    "ICU"  : "intensive care unit",
    "DOB"  : "date of birth",
    "BMI"  : "body mass index",
    "GCS"  : "glasgow coma scale",
    "ECG"  : "electrocardiogram",
    "MRI"  : "magnetic resonance imaging",
}

STOPWORDS = {
    "patient","was","the","and","with","of","to","a","in","is",
    "were","for","on","at","be","has","had","this","that","he",
    "she","her","his","we","as","an","by","are","been","also",
    "which","from","or","but","no","not","it","its","than","then",
}

def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for abbr, full in CLINICAL_ABBR.items():
        text = re.sub(rf"\b{abbr.lower()}\b", full, text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────
#  ENTITY EXTRACTORS  (regex per category)
# ─────────────────────────────────────────────────────────
EXTRACTORS = {
    "Demographics": {
        "Name":        r"\b([A-Z][a-z]{1,15} [A-Z][a-z]{1,15})\b(?!\s+(?:mmHg|bpm|year|old|mg|ml))",
        "Age":         r"\b(\d{1,3})\s*[-]?\s*year[s]?\s*old",
        "Gender":      r"\b(male|female|man|woman|boy|girl)\b",
        "DOB":         r"\b(?:dob|date of birth)[:\s]+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
        "Patient ID":  r"\b(?:patient id|id no|mrn)[:\s#]+([A-Z0-9\-]+)",
        "Occupation":  r"\b(?:occupation|works as|employed as)[:\s]+([a-zA-Z\s]+?)(?:\.|,|$)",
        "Marital":     r"\b(married|unmarried|single|divorced|widowed)\b",
        "Blood Group": r"blood group[:\s]+([ABO]{1,2}[+-])",
        "Admitted":    r"admitted\s+(?:on\s+)?(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})",
    },
    "Vital Signs": {
        "Blood Pressure":   r"(?:BP|blood pressure)\s*(?:was|:)?\s*(\d{2,3}\/\d{2,3})\s*(?:mmHg)?",
        "Heart Rate":       r"(?:heart rate|HR|pulse)\s*(?:was|:)?\s*(\d{2,3})\s*(?:bpm)?",
        "Temperature":      r"temperature\s*(?:was|:)?\s*(\d{2,3}\.?\d*)\s*(?:F|C|°)",
        "SpO2":             r"(?:SpO2|oxygen saturation)\s*(?:was|:)?\s*(\d{2,3})\s*%?",
        "Respiratory Rate": r"(?:respiratory rate|RR)\s*(?:was|:)?\s*(\d{1,2})\s*(?:per min|\/min)?",
        "BMI":              r"BMI\s*(?:was|:)?\s*(\d{2,3}\.?\d*)",
        "Weight":           r"weight\s*(?:was|:)?\s*(\d{2,3})\s*kg",
        "Height":           r"height\s*(?:was|:)?\s*(\d{3})\s*cm",
    },
    "Laboratory Reports": {
        "Hemoglobin":  r"(?:hemoglobin|Hgb)\s*(?:was|:)?\s*(\d+\.?\d*)\s*(?:g\/dL)?",
        "WBC":         r"WBC\s*(?:was|:)?\s*(\d{3,6})",
        "Platelets":   r"platelet\s*(?:count|:)?\s*(\d{3,6})",
        "HbA1c":       r"HbA1c\s*(?:was|:)?\s*(\d+\.?\d*)\s*%?",
        "Creatinine":  r"creatinine\s*(?:was|:)?\s*(\d+\.?\d*)",
        "Glucose":     r"(?:glucose|blood sugar)\s*(?:was|:)?\s*(\d+\.?\d*)\s*(?:mg\/dL)?",
        "Cholesterol": r"(?:cholesterol|LDL|HDL)\s*(?:was|:)?\s*(\d+\.?\d*)",
        "Sodium":      r"(?:sodium|Na)\s*(?:was|:)?\s*(\d+\.?\d*)",
        "Potassium":   r"(?:potassium|K)\s*(?:was|:)?\s*(\d+\.?\d*)",
    },
    "Pathology Reports": {
        "Specimen":  r"(?:biopsy|specimen|tissue)\s+(?:of\s+)?(\w+\s?\w*)\s+(?:shows?|reveals?)",
        "Diagnosis": r"\b(carcinoma|adenocarcinoma|sarcoma|lymphoma|dysplasia|fibrosis|hepatitis|malignancy)\b[\w\s]*",
        "Grade":     r"(?:grade|gleason\s+score)\s*[:\s]*([IVX1-9]+)",
        "Stage":     r"stage\s+([A-Z0-9]+)",
        "Margin":    r"margins?\s+(?:are\s+)?(clear|involved|positive|negative)",
    },
    "Microbiology Data": {
        "Organism":    r"\b(Staphylococcus[\w\s]*|E\.?\s*coli|Klebsiella[\w\s]*|Pseudomonas[\w\s]*|Salmonella[\w\s]*|Candida[\w\s]*|Streptococcus[\w\s]*|MRSA|H1N1|Tuberculosis|AFB)\b",
        "Culture":     r"(blood|urine|sputum|wound|stool|CSF|throat)\s+(?:culture|swab)",
        "CFU":         r"(\d+)\s*CFU",
        "Resistance":  r"resistant\s+(?:to\s+)?([\w\s]+?)(?:\.|,|sensitive|$)",
        "Sensitivity": r"sensitive\s+(?:to\s+)?([\w\s]+?)(?:\.|,|resistant|$)",
        "Test Result": r"(PCR|antigen|smear|AFB)\s+(positive|negative)",
    },
    "Clinical Notes": {
        "Complaint":   r"(?:complains?\s+of|presents?\s+with|history\s+of)\s+([\w\s,]+?)(?:\.|,\s+(?:for|since)|for\s+\d|$)",
        "Duration":    r"(?:for|since|past)\s+(\d+\s+(?:days?|weeks?|months?|hours?))",
        "Examination": r"on\s+examination[,:\s]+([\w\s,]+?)(?:\.|$)",
        "Impression":  r"(?:impression|assessment)[:\s]+([\w\s,]+?)(?:\.|$)",
        "Plan":        r"(?:referred\s+to|review\s+after|advised\s+to)\s+([\w\s,]+?)(?:\.|,|$)",
        "Medication":  r"(?:started|given|prescribed|administered)\s+([\w]+(?:\s+[\w]+)?)",
    },
}

# ─────────────────────────────────────────────────────────
#  RULE-BASED BOOSTERS
# ─────────────────────────────────────────────────────────
RULE_SIGNALS = {
    "Demographics": [
        r"\b\d{1,3}\s*year\s*old\b",
        r"\b(male|female)\b",
        r"\b(married|single|widowed)\b",
        r"\badmitted\s+on\b",
        r"\bdate\s+of\s+birth\b",
        r"\bblood\s+group\b",
        r"\boccupation\b",
    ],
    "Vital Signs": [
        r"\b\d{2,3}\/\d{2,3}\s*mmHg\b",
        r"\bheart\s+rate\b",
        r"\bblood\s+pressure\b",
        r"\bSpO2\b",
        r"\b\d{2,3}\s*bpm\b",
        r"\btemperature\s+\d",
        r"\bpulse\b",
    ],
    "Laboratory Reports": [
        r"\bhemoglobin\b",
        r"\bWBC\b",
        r"\bplatelet\b",
        r"\bHbA1c\b",
        r"\bcreatinine\b",
        r"\bg\/dL\b",
        r"\bmg\/dL\b",
    ],
    "Pathology Reports": [
        r"\bbiopsy\b",
        r"\bcarcinoma\b",
        r"\badenocarcinoma\b",
        r"\bhistopathology\b",
        r"\bdysplasia\b",
        r"\bgleason\b",
    ],
    "Microbiology Data": [
        r"\bculture\b",
        r"\bCFU\b",
        r"\bresistant\s+to\b",
        r"\bsensitive\s+to\b",
        r"\bPCR\b",
        r"\bAFB\b",
        r"\bMRSA\b",
    ],
    "Clinical Notes": [
        r"\bcomplains?\s+of\b",
        r"\bpresents?\s+with\b",
        r"\bon\s+examination\b",
        r"\breferred\s+to\b",
        r"\bdischarge\b",
        r"\bimpression\b",
        r"\bplan\b",
    ],
}

BOOST_THRESHOLD = 15.0
RULE_BOOST      = 25.0


# ─────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────
def load_and_train(csv_path="mtsamples.csv"):
    df_raw = pd.read_csv(csv_path)
    df_raw["category"] = df_raw["medical_specialty"].str.strip().map(CATEGORY_MAP)
    df_raw["text"] = (
        df_raw["description"].fillna("") + " " +
        df_raw["transcription"].fillna("") + " " +
        df_raw["keywords"].fillna("")
    )
    df = df_raw[["text","category"]].dropna().reset_index(drop=True)
    df["clean_text"] = df["text"].apply(preprocess)
    df = df[df["clean_text"].str.split().apply(len) >= 10]

    dfs = []
    for cat in df["category"].unique():
        subset = df[df["category"] == cat]
        if len(subset) < 20:
            continue
        dfs.append(resample(subset, replace=len(subset) < 600,
                            n_samples=600, random_state=42))
    df_bal = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    le = LabelEncoder()
    X  = df_bal["clean_text"].tolist()
    y  = le.fit_transform(df_bal["category"])
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    lr  = LogisticRegression(C=10, max_iter=3000, solver="saga",
                              multi_class="multinomial", random_state=42, n_jobs=-1)
    svm = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=3000, random_state=42))

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,3), max_features=20000,
                                   sublinear_tf=True, min_df=2, max_df=0.90)),
        ("clf",   VotingClassifier(estimators=[("lr",lr),("svm",svm)],
                                   voting="soft", weights=[2,1])),
    ])
    model.fit(X_train, y_train)
    return model, le


# ─────────────────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────────────────
def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    expanded  = []
    for s in sentences:
        parts = [p.strip() for p in s.split(",") if len(p.strip()) > 10]
        expanded.extend(parts)
    return [s for s in expanded if len(s.split()) >= 3]


def multilabel_predict(raw_text, model, le, threshold=BOOST_THRESHOLD):
    sentences = split_sentences(raw_text)
    if not sentences:
        sentences = [raw_text]

    details = defaultdict(lambda: {"avg_confidence": 0.0, "detected_in": 0, "sentences": []})

    for sent in sentences:
        cleaned = preprocess(sent)
        if not cleaned or len(cleaned.split()) < 2:
            continue
        pred_idx = model.predict([cleaned])[0]
        proba    = model.predict_proba([cleaned])[0]
        conf_dict = {cls: round(float(p)*100, 1) for cls, p in zip(le.classes_, proba)}

        for cat, conf in conf_dict.items():
            if conf >= threshold:
                prev = details[cat]
                n    = prev["detected_in"]
                prev["avg_confidence"] = round((prev["avg_confidence"]*n + conf)/(n+1), 1)
                prev["detected_in"]   += 1
                prev["sentences"].append(sent)

    # Rule-based boosting
    for cat, patterns in RULE_SIGNALS.items():
        hits = sum(1 for p in patterns if re.search(p, raw_text, re.IGNORECASE))
        if hits > 0:
            boost = hits * (RULE_BOOST / len(patterns)) * 100
            if cat in details:
                details[cat]["avg_confidence"] = min(99.0, details[cat]["avg_confidence"] + boost)
            else:
                details[cat] = {"avg_confidence": min(99.0, boost),
                                "detected_in": hits, "sentences": [raw_text]}

    details = {k: v for k, v in details.items() if v["avg_confidence"] >= threshold}
    details = dict(sorted(details.items(), key=lambda x: x[1]["avg_confidence"], reverse=True))

    return {"detected_labels": list(details.keys()), "label_details": details}


def extract_entities(text, category):
    patterns  = EXTRACTORS.get(category, {})
    extracted = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = next((g for g in match.groups() if g), None)
            if value:
                extracted[field] = value.strip()
    return extracted


def structured_classify(raw_text, model, le):
    result   = multilabel_predict(raw_text, model, le)
    detected = result["detected_labels"]
    output   = {}
    for cat in detected:
        evidence = " ".join(result["label_details"][cat]["sentences"])
        combined = evidence + " " + raw_text
        entities = extract_entities(combined, cat)
        output[cat] = {
            "confidence": result["label_details"][cat]["avg_confidence"],
            "entities":   entities,
        }
    return output
