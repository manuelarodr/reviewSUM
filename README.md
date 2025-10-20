# Credibility-Aware Review Summarizer

A transparent, credibility-aware product review summarizer that emphasizes interpretability, user control, and traceability.

## Features

- Generate summaries that highlight both majority and minority perspectives
- Display statistical insights such as "79% mention durability, 8% cite safety issues"
- Allow users to filter by credibility (verified-only, helpfulness threshold)
- Ensure every summary is traceable back to its source reviews
- Evaluate factual consistency against human-written website summaries

## Project Structure

```
reviewSUM/
├── src/
│   ├── filtering.py      # Filter low-credibility reviews
│   ├── themes.py         # Extract product themes/attributes
│   ├── summarizer.py     # Chain-of-Density summarization
│   ├── evaluation.py     # ROUGE, factual consistency, coverage
│   ├── ui/
│   │   └── dashboard.py  # Interactive Streamlit interface
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py
│       └── config.py
├── data/
│   └── amasum-5productsample/  # Sample dataset
├── tests/
│   ├── test_filtering.py
│   ├── test_themes.py
│   ├── test_summarizer.py
│   └── test_evaluation.py
├── requirements.txt
├── README.md
└── main.py
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download NLTK data: `python -c "import nltk; nltk.download('punkt')"`

## Usage

### Command Line Interface
```bash
python main.py --input data/amasum-5productsample/0062820680.json --output summary.json
```

### Interactive Dashboard
```bash
streamlit run src/ui/dashboard.py
```

## Core Modules

### 1. Filtering Module (`filtering.py`)
Filters out low-credibility reviews using metadata (verified, helpful_votes) and text heuristics.

### 2. Themes Module (`themes.py`)
Extracts product themes or attributes using embeddings or topic models.

### 3. Summarizer Module (`summarizer.py`)
Implements Chain-of-Density summarization using LLM with Groq API or local transformers.

### 4. Evaluation Module (`evaluation.py`)
Computes ROUGE scores, factual consistency, and coverage metrics.

### 5. UI Module (`ui/dashboard.py`)
Interactive Streamlit interface to explore summaries and filter by credibility.

## Dataset Format

The system expects JSON files with the following structure:

```json
{
  "website_summaries": [
    {
      "verdict": "Human-written summary",
      "pros": ["pro1", "pro2"],
      "cons": ["con1", "con2"],
      "source": "source_name"
    }
  ],
  "customer_reviews": [
    {
      "title": "Review title",
      "text": "Review content",
      "rating": 5.0,
      "verified": true,
      "author": "Author name",
      "helpful_votes": 79,
      "publication_date": 20180501
    }
  ],
  "product_meta": {
    "title": "Product title",
    "rating": 4.5,
    "categories": ["category1", "category2"]
  }
}
```

## Configuration

Create a `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black src/ tests/
```

Lint code:
```bash
flake8 src/ tests/
```
