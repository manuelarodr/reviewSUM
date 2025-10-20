import os
import json
import pandas as pd

# Path to the folder containing product review files
PRODUCTS_FOLDER = "amasum-5productsample"

# Get all JSON files in the folder
product_files = [f for f in os.listdir(PRODUCTS_FOLDER) if f.endswith('.json')]

# Prepare table data
rows = []
for filename in product_files:
    filepath = os.path.join(PRODUCTS_FOLDER, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Get product title
    product_title = data.get('product_meta', {}).get('title', filename.replace('.json', ''))
    # Get reviews list
    reviews = data.get('customer_reviews', [])
    total_reviews = len(reviews)
    valid_reviews = [r for r in reviews if isinstance(r, dict) and 'rating' in r and isinstance(r['rating'], (int, float))]
    star_counts = {star: sum(1 for r in valid_reviews if int(r['rating']) == star) for star in range(1, 6)}
    rows.append({
        'Product': product_title,
        'Total Reviews': total_reviews,
        '1 Star': star_counts[1],
        '2 Stars': star_counts[2],
        '3 Stars': star_counts[3],
        '4 Stars': star_counts[4],
        '5 Stars': star_counts[5],
    })

# Create and print table
stats_df = pd.DataFrame(rows)
print(stats_df.to_markdown(index=False))
