from pydantic import BaseModel

class AnonymizationKeys(BaseModel):
    remove: list[str] = []
    empty: list[str] = []
    hash: list[str] = []
    increment_date: list[str] = []