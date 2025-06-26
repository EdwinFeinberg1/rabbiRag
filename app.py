import os
from flask import Flask, request, jsonify, render_template
import openai
from rabbi_rag import RabbiRAG

# Derech Hashem by Ramchal + The Beginning of Wisdom
BOOKS = [
    # Derekh Hashem chapters
    "Derekh_Hashem,_Part_One,_On_the_Creator",
    "Derekh_Hashem,_Part_One,_On_the_Purpose_of_Creation",
    "Derekh_Hashem,_Part_One,_On_Mankind",
    "Derekh_Hashem,_Part_One,_On_Human_Responsibility",
    "Derekh_Hashem,_Part_One,_On_the_Spiritual_Realm",
    "Derekh_Hashem,_Part_Two,_On_Divine_Providence_in_General",
    "Derekh_Hashem,_Part_Two,_On_Mankind_in_This_World",
    "Derekh_Hashem,_Part_Two,_On_Personal_Providence",
    "Derekh_Hashem,_Part_Two,_On_Israel_and_the_Nations",
    "Derekh_Hashem,_Part_Three,_On_the_Soul_and_Its_Activities",
    "Derekh_Hashem,_Part_Three,_On_Divine_Names_and_Witchcraft",
    "Derekh_Hashem,_Part_Three,_On_Divine_Inspiration_and_Prophecy",
    "Derekh_Hashem,_Part_Four,_On_Divine_Service",
    "Derekh_Hashem,_Part_Four,_On_Torah_Study",
    "Derekh_Hashem,_Part_Four,_On_Love_and_Fear_of_God",
    "Derekh_Hashem,_Part_Four,_On_Prayer",
    # The Beginning of Wisdom
    "The_Beginning_of_Wisdom"
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
