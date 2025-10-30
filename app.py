from flask import Flask, request, render_template
from main import load_and_train_model, predict_message

app = Flask(__name__)

# Load the model and vectorizer
model, vectorizer, feature_columns = load_and_train_model()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form.get("message")
        if user_input:
            result = predict_message(user_input, model, vectorizer, feature_columns)
            return render_template("index.html", prediction=result, message=user_input)
    return render_template("index.html", prediction=None, message=None)

if __name__ == "__main__":
    app.run(debug=True)