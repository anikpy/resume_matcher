import io
import re
import PyPDF2
import spacy
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from langdetect import detect
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")

en_nlp = spacy.load("en_core_web_sm")
de_nlp = spacy.load('de_core_news_lg')


EMAIL_PATTERN = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
PHONE_PATTERN = r"\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}"
EXPERIENCE_PATTERN = r"(experience|work history|employment)\s*[:\-]?\s*(.*?)(?=\n[A-Z]|$)"
EDUCATION_PATTERN = r"(bachelor|master|phd)\s*(of)?\s*([a-z\s]+)"
SKILL_SECTION_PATTERN = r"(skills|technical skills|proficiencies)\s*[:\-]?\s*(.*?)(?=\n[A-Z]|$)"
EXPERIENCE_SECTION_PATTERN = r"(?:PROFESSIONAL EXPERIENCE|Experience|Work Experience|Employment)\s*(.*?)(?=\n *(?:Education|Achievement|References|Skills|EXPERTISE|$))"
JOB_PATTERN = r"((?:[A-Z][a-zA-Z\s,]+?))\s*\n\s*([A-Za-z\s\(\)]+?)(?:\s*at\s*|\s*,\s*|\s*–\s*|\s*\n\s*)([A-Za-z\s\d,-]+?)\s*\n\s*((?:October|November|January|February|March|April|May|June|July|August|September|December)?\s*\d{4}\s*[-–]\s*(?:(?:October|November|January|February|March|April|May|June|July|August|September|December)?\s*\d{4}|Present)\s*(?:\((.*?)\))?)"


def detect_langauge(text: str) -> str:
    try:
        lang = detect(text)
        return lang
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error detecting language: {str(e)}")

def cv_extract_pdf_text(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CV PDF: {str(e)}")


def cv_extract_skills(cv_text: str) -> List[str]:
    cv_text_lower = cv_text.lower()
    cv_doc = en_nlp(cv_text_lower)
    skills = set()
    skill_matches = re.finditer(SKILL_SECTION_PATTERN, cv_text, re.DOTALL | re.IGNORECASE)
    for match in skill_matches:
        skill_text = match.group(2).strip()
        skill_lines = skill_text.split("\n")
        for line in skill_lines:
            line = line.strip()
            if line:
                line_doc = en_nlp(line)
                for chunk in line_doc.noun_chunks:
                    if len(chunk.text.split()) <= 3:
                        skills.add(chunk.text.strip())
    for chunk in cv_doc.noun_chunks:
        if len(chunk.text.split()) <= 3 and any(token.pos_ in ["NOUN", "PROPN"] for token in chunk):
            skills.add(chunk.text.strip())
    return sorted(skills)



