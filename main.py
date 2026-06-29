from typing import List
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from langdetect import detect
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import spacy
import re
import logging
from sklearn.metrics.pairwise import cosine_similarity
from functions import connect_to_elasticsearch, extract_job_skills

en_nlp = spacy.load("en_core_web_sm")
de_nlp = spacy.load("de_core_news_lg")

app = FastAPI()


es = connect_to_elasticsearch()
model = SentenceTransformer('BAAI/bge-m3')
nlp = spacy.load("en_core_web_sm")
INDEX_NAME = "cvdata"

@app.post("/CvMatcher")
async def match_cv(request: Request):
    try:
        data = await request.json()
        job_text = data.get("job_text")

        if not job_text:
            raise HTTPException(status_code=400, detail="Missing 'job_text' in request.")

        extracted_skills = extract_job_skills(job_text)
        skills_str = " ".join(extracted_skills)
        job_embedding = model.encode(skills_str)
        job_vector = np.array(job_embedding).reshape(1, -1)

        # query = {
        #     "size": 5,
        #     "query": {
        #         "script_score": {
        #             "query": {"match_all": {}},
        #             "script": {
        #                 "source": """
        #                             cosineSimilarity(params.job_vector, 'embedding') +
        #                             cosineSimilarity(params.skill_vector, 'skill_embedding')
        #                         """,
        #                 "params": {
        #                     "job_vector": job_embedding.tolist(),
        #                     "skill_vector": skill_embedding.tolist()
        #                 }
        #             }
        #         }
        #     }
        # }

        response = es.search(
            index=INDEX_NAME,
            body={
                "query": {
                    "match_all": {}
                },
                "_source": ["filename", "embedding", "skill_embedding"]
            },
            size=100  # Adjust as needed
        )
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            cv_id = hit["_id"]
            file_name = source.get("filename", "unknown")
            doc_embedding = np.array(source.get("embedding", []))
            skill_embedding = np.array(source.get("skill_embedding", []))
            if doc_embedding.size == 0 or skill_embedding.size == 0:
                continue
            doc_sim = cosine_similarity(job_vector, doc_embedding.reshape(1, -1))[0][0]
            skill_sim = cosine_similarity(job_vector, skill_embedding.reshape(1, -1))[0][0]
            avg_sim = (doc_sim + skill_sim) / 2
            match_percentage = round(avg_sim * 100, 2)
            results.append({
                "cv_id": cv_id,
                "filename": file_name,
                "match_percentage": match_percentage
            })
        results.sort(key=lambda x: x["match_percentage"], reverse=True)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
