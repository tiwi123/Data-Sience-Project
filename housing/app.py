from fastapi import FastAPI
from pydantic import BaseModel
import requests
import model
import database
import uvicorn

app = FastAPI()

# Model untuk input properti
class PropertyInput(BaseModel):
    total_bedrooms: int
    total_rooms: int
    median_income: float
    ocean_proximity: str

# Endpoint untuk prediksi harga rumah
@app.post("/predict")
def predict_price(property: PropertyInput):
    try:
        user_data = requests.get("https://randomuser.me/api/").json()
        user_id = user_data['results'][0]['login']['uuid']

        predicted_price = model.predict(
            property.total_bedrooms,
            property.total_rooms,
            property.median_income,
            property.ocean_proximity
        )

        database.save_to_db(user_id, property.dict(), predicted_price)

        return {"user_id": user_id, "predicted_price": predicted_price}
    except Exception as e:
        return {"error": str(e)}

# Endpoint untuk mendapatkan statistik harga rumah
@app.get("/stats")
def get_stats():
    stats = model.get_statistics()
    return stats

# Menjalankan aplikasi
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
