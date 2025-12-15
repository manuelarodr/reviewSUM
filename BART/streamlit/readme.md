Based on David's bart.py file, I created summarizer.py to summarize reviews for individual products. I think it saves runtime and is more suitable for presentation purposes. The original bart.py output put analysis results into a CSV file. I disabled CSV generation so that results are now displayed on the web (or we gonna have a bunch of csv files).

How to use:
Launch Streamlit, open the webpage, and upload the JSON file (Note: It must contain reviews for a single product. I used the JSON file from the "amasum-5productsample" folder and it works properly).
