import os
from flask import Flask, request, jsonify, render_template
import openai
from rabbi_rag import RabbiRAG

# Choose five books with English translations
BOOKS = [
    "Genesis",
    "Exodus",
    "Leviticus",
    "Numbers",
    "Deuteronomy",
]

rag = RabbiRAG(BOOKS)
rag.build()

openai.api_key = os.environ.get("OPENAI_API_KEY")


def call_openai(prompt):
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message["content"].strip()


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    answer, citations = rag.answer(question, call_openai)
    return jsonify({"answer": answer, "citations": citations})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
