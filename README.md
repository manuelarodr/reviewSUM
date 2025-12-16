# ReviewSUM

## Project Structure

- **src/** - Main project code including the dashboard UI, summarization engine, and Q&A.
- **amasum-5productsample/** - 5 examples of AMASUM product review JSONs (from https://github.com/abrazinskas/SelSum/tree/master/data)
- **BART/** - Archived code from the BART exploration phase


## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd reviewSUM
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your GROQ API key:**
   - Create a `.env` file in the root directory
   - Add your GROQ API key: `GROQ_API_KEY=your_api_key_here`

## Running the Dashboard

Launch the interactive dashboard:

```bash
streamlit run src/ui/dashboard.py
```

The application will open at `http://localhost:8501`
