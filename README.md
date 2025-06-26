# RabbiGPT

RabbiGPT is a minimal Retrieval Augmented Generation (RAG) interface that uses a
subset of texts from [Sefaria](https://www.sefaria.org/) to answer questions.
Only five books of the Torah are used (Genesis, Exodus, Leviticus, Numbers and
Deuteronomy) and all text is pulled from the English translation available on
Sefaria.

The application consists of a small Flask server with a simple mobile-friendly
front end. Queries are embedded with `sentence-transformers` and stored in a
FAISS index. Answers are generated using the OpenAI API with the relevant
passages passed as context. Each answer includes links to the original text on
Sefaria.

## Usage

1. Install requirements

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Set your OpenAI API key

   ```bash
   export OPENAI_API_KEY=your-key
   ```

3. Run the app

   ```bash
   python app.py
   ```

4. Visit `http://localhost:8000` on your mobile device or simulator.

## Notes

- On first run, the server will fetch the five books from Sefaria's API to
  build the local index.
- Network access is required for downloading the texts and for OpenAI API calls.
- The RAG pipeline returns a list of citations with direct links to Sefaria for
  each passage used in the answer.
