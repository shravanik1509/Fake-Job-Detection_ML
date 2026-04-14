import pickle

# Load model and vectorizer
model = pickle.load(open("fake_job_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_job(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)

    if result[0] == 1:
        return "Fake Job ❌"
    else:
        return "Real Job ✅"

# Test example
if __name__ == "__main__":
    sample = "Earn money from home easily without experience"
    print("Prediction:", predict_job(sample))
