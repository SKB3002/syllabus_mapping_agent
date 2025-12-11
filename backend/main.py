from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import base64
import uvicorn
import shutil
from curriculum_comparator import CurriculumComparator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process(
    boardA: str = Form(...),
    boardB: str = Form(...),
    fileA: UploadFile = Form(...),
    fileB: UploadFile = Form(...)
):

    pathA = f"tempA.{fileA.filename.split('.')[-1]}"
    pathB = f"tempB.{fileB.filename.split('.')[-1]}"

    with open(pathA, "wb") as f:
        shutil.copyfileobj(fileA.file, f)

    with open(pathB, "wb") as f:
        shutil.copyfileobj(fileB.file, f)


    comp = CurriculumComparator(pathA, pathB, boardA_name=boardA, boardB_name=boardB)
    comp.load_data()
    comp.preprocess()
    comp.build_combined_text()
    comp.generate_embeddings()
    comp.compute_embedding_similarity()
    comp.compute_topic_similarity()
    comp.extract_concepts()
    comp.compute_concept_similarity_matrix()
    comp.cluster_embeddings()
    comp.run_agreement_engine()
    comp.export_results("comparison_output.csv")

    df = pd.read_csv("comparison_output.csv")


    csv_b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()

    return JSONResponse({
        "preview": df.head(10).to_dict(orient="records"),
        "csv_b64": csv_b64
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
