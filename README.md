
Final product lives in src
BART folder contains our initial attempt at the summaries with BART


To run the dashboard.py in src\ui:

1. Install Python 3.10+ and create/activate a virtual env.

2. From the repo root, install dependencies: pip install -r requirements.txt.

3. Set your Groq API key: create .env in the repo root with GROQ_API_KEY=your_key_here.

4. Start the dashboard: streamlit run src/ui/dashboard.py.

5. In the app, upload an AMASUM-format product JSON to generate summaries and use Q&A.
