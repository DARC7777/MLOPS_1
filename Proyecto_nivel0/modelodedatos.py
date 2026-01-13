from pydantic import BaseModel, Field
from typing import List

class Tablero(BaseModel):
    # Validamos que sea una lista de 784 números (28*28)
    # Los valores de píxeles suelen ser entre 0 (blanco) y 255 (negro)
    pixeles: List[float] = Field(..., min_items=784, max_items=784)