# import io
# import os
# import re
# import PyPDF2
# import spacy
# import warnings
# from typing import List
# import textacy
# from textacy import extract
# from sentence_transformers import SentenceTransformer, util
# from langdetect import detect
# warnings.filterwarnings("ignore", message="Can't initialize NVML")
#
# en_nlp = spacy.load("en_core_web_sm")
# de_nlp = spacy.load("de_core_news_lg")
# model = SentenceTransformer('BAAI/bge-m3')
#
# SKILL_SECTION_PATTERN = r"(skills|technical skills|proficiencies|kenntnisse|fähigkeiten|kompetenzen)\s*[:\-]?\s*(.*?)(?=\n[A-ZÄÖÜ]|$)"
#
# def extract_pdf_text(pdf_file: bytes) -> str:
#     reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
#     return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
#
# def clean_text(text: str) -> str:
#     text = re.sub(r'•|\n+', ' ', text)
#     return re.sub(r'\s+', ' ', text).strip()
#
# def detect_language(text: str) -> str:
#     try:
#         return detect(text)
#     except Exception:
#         return "en"
#
#
# def extract_skills_by_language(text: str, lang: str) -> List[str]:
#     nlp = en_nlp if lang == "en" else de_nlp
#     doc = nlp(text.lower())
#     skills = set()
#
#     for match in re.finditer(SKILL_SECTION_PATTERN, text, re.DOTALL | re.IGNORECASE):
#         for line in match.group(2).strip().split("\n"):
#             line_doc = nlp(line.strip())
#             noun_chunks = textacy.extract.noun_chunks(line_doc, min_freq=1)
#             for chunk in noun_chunks:
#                 if len(chunk.text.split()) <= 3:
#                     skills.add(chunk.text.strip())
#
#     noun_chunks = textacy.extract.noun_chunks(doc, min_freq=1)
#     for chunk in noun_chunks:
#         if len(chunk.text.split()) <= 3 and any(token.pos_ in ["NOUN", "PROPN"] for token in chunk):
#             skills.add(chunk.text.strip())
#
#     return sorted(skills)
#
# def extract_job_skills(text: str, skill_lexicon: List[str] = None) -> List[str]:
#     lang = detect_language(text)
#     text = clean_text(text.lower())
#     nlp = en_nlp if lang == "en" else de_nlp
#     doc = nlp(text)
#     skills = set()
#     if skill_lexicon:
#         for skill in skill_lexicon:
#             if skill.lower() in text:
#                 skills.add(skill)
#     for chunk in doc.noun_chunks:
#         chunk_text = chunk.text.strip()
#         if (len(chunk_text.split()) <= 4 and any(t.pos_ in ["NOUN", "PROPN"] for t in chunk)
#                 and not any(w in chunk_text for w in ["our", "your", "the", "unsere", "ihre", "der"]) and len(chunk_text) > 2):
#             skills.add(chunk_text)
#     return sorted(skills)
#
# def calculate_similarity(cv_skills: List[str], job_skills: List[str]) -> float:
#     cv_embeddings = model.encode(cv_skills, convert_to_tensor=True)
#     job_embeddings = model.encode(job_skills, convert_to_tensor=True)
#     similarity_matrix = util.cos_sim(cv_embeddings, job_embeddings)
#     best_similarities = similarity_matrix.max(dim=1).values
#     return round(best_similarities.mean().item() * 100, 2)
#
# job_text = extract_pdf_text(open('Data/description/description.pdf', 'rb').read())
# job_skills = extract_job_skills(job_text)
#
# cv_folder = 'Data/cv'
# best_match = None
# highest_score = 0.0
#
# for filename in os.listdir(cv_folder):
#     if filename.endswith('.pdf'):
#         cv_path = os.path.join(cv_folder, filename)
#         cv_text = extract_pdf_text(open(cv_path, 'rb').read())
#         lang = detect_language(cv_text)
#         cv_skills = extract_skills_by_language(cv_text, lang)
#         score = calculate_similarity(cv_skills, job_skills)
#         print(f"{filename} → Language: {lang} → Similarity: {score}%")
#         if score > highest_score:
#             highest_score = score
#             best_match = filename
#
# print(f"\nBest matching CV: {best_match} ({highest_score}%)")
#
