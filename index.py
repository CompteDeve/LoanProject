from fastapi import FastAPI
app = FastAPI()

@app.get("/api/python")
def hello():
    return {"message": "Le Backend Python tourne sur Vercel !"}