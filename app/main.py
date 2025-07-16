from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pick_regno import pick_regno
import traceback


MODEL_PATH = 'micromodel.cbm'
app = FastAPI()

class InputData(BaseModel):
    camera_regno: str
    nn_regno: str
    camera_score: float
    nn_score: float
    nn_sym_scores: str
    nn_len_scores: str
    camera_type: str
    camera_class: str
    time_check: str
    direction: str

@app.post('/predict')
def predict(data: InputData):
    try:
        result = pick_regno(
            data.regno_recognize,
            data.nn_regno,
            data.camera_score,
            data.nn_score,
            data.nn_sym_scores,
            data.nn_len_scores,
            data.camera_type,
            data.camera_class,
            data.time_check,
            data.direction,
            MODEL_PATH
        )
        return {'prediction': result.tolist()}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


