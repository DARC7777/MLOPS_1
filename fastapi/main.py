from fastapi import FastAPI
import metodos
import modelodedatos
import respuestayestados

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello, FastAPI!"}
    s