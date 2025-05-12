import pandas as pd
import re
import nltk
import math
from collections import defaultdict

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')  # Added missing resource

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Initialize NLP components
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Load and prepare data
try:
    df = pd.read_csv(r'Data\GoodReads_100k_books.csv')
    # Clean and validate titles
    titles = [str(title) for title in df['title'].dropna().unique() if str(title).strip()]
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    exit()

# Enhanced text preprocessing
def preprocess(text):
    try:
        text = text.lower().strip()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        tokens = [stemmer.stem(lemmatizer.lemmatize(t)) for t in tokens]
        return tokens
    except:
        return []

# Build inverted index
inverted_index = defaultdict(set)
for idx, title in enumerate(titles):
    tokens = preprocess(title)
    for token in tokens:
        inverted_index[token].add(idx)

# TF-IDF calculation with normalization
def compute_tf_idf(docs):
    tf_idf = []
    doc_freq = defaultdict(int)
    total_docs = len(docs)
    
    # Document frequency calculation
    for doc in docs:
        unique_tokens = set(preprocess(doc))
        for token in unique_tokens:
            doc_freq[token] += 1
    
    # TF-IDF vectorization
    for doc in docs:
        tf = defaultdict(float)
        tokens = preprocess(doc)
        doc_length = len(tokens) or 1
        
        for token in tokens:
            tf[token] += 1
        
        # Normalize and calculate TF-IDF
        tfidf = {}
        for token, count in tf.items():
            tf_val = count / doc_length
            idf_val = math.log((total_docs + 1) / (doc_freq.get(token, 0) + 1))
            tfidf[token] = tf_val * idf_val
        
        tf_idf.append(tfidf)
    
    return tf_idf

tf_idf = compute_tf_idf(titles)

# Search function with ranking
def search(query, titles, tf_idf, top_n=100):
    query_tokens = preprocess(query)
    scores = []
    
    for doc_idx, doc_vector in enumerate(tf_idf):
        score = sum(doc_vector.get(token, 0) for token in query_tokens)
        scores.append((score, doc_idx))
    
    # Sort by score and index
    ranked_results = sorted(scores, key=lambda x: (-x[0], x[1]))
    return [(idx, titles[idx]) for score, idx in ranked_results[:top_n]]

# Evaluation metrics calculator
def evaluate(retrieved, relevant, selected_positions, top_displayed=10):
    displayed = retrieved[:top_displayed]
    tp = len(selected_positions)
    fp = top_displayed - tp
    
    # Precision and Recall calculations
    precision = tp / (tp + fp + 1e-10)
    relevant_in_retrieved = len(set(retrieved) & set(relevant))
    recall = relevant_in_retrieved / (len(relevant) + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return precision, recall, f1

# Find potential relevant documents
def find_potentially_relevant(query, titles, exclude_ids):
    query_tokens = preprocess(query)
    candidates = []
    
    for idx, title in enumerate(titles):
        if idx not in exclude_ids:
            title_tokens = preprocess(title)
            if any(token in title_tokens for token in query_tokens):
                candidates.append((idx, title))
    
    return candidates

# Main application flow
if __name__ == "__main__":
    print("\n" + "="*60)
    print(" Advanced Book Search Engine - GoodReads Edition")
    print("="*60)
    
    # Get user query
    query = input("\n🔍 Enter your search query: ").strip()
    while not query:
        print(" Please enter a valid search query.")
        query = input("\n🔍 Enter your search query: ").strip()
    
    # Perform search
    results = search(query, titles, tf_idf)
    
    # Display top results
    print(f"\n Top 10 Results for: '{query}'")
    for rank, (idx, title) in enumerate(results[:10], 1):
        print(f"{rank:2d}. [{idx}] {title}")
    
    # Relevance feedback
    print("\n STEP 1: Mark relevant results (1-10)")
    selected = list(map(int, input("Comma-separated positions (e.g., 1,3): ").strip().split(',')))
    selected = [p for p in selected if 1 <= p <= 10]
    
    # Calculate metrics
    displayed_ids = [idx for idx, _ in results[:10]]
    relevant_ids = [displayed_ids[p-1] for p in selected]
    retrieved_ids = [idx for idx, _ in results[:100]]
    
    # Find missed relevant documents
    print("\n Searching for potentially missed relevant books...")
    candidates = find_potentially_relevant(query, titles, retrieved_ids)
    
    # User validation of candidates
    false_negatives = []
    if candidates:
        print("\n STEP 2: Identify any relevant books from these candidates:")
        for i, (idx, title) in enumerate(candidates[:5], 1):
            print(f"{i}. [{idx}] {title}")
        
        selections = list(map(int, input("\nEnter positions (comma-separated): ").strip().split(',')))
        false_negatives = [candidates[p-1][0] for p in selections if 1 <= p <= len(candidates[:5])]
    
    all_relevant = relevant_ids + false_negatives
    
    # Final evaluation
    if all_relevant:
        precision, recall, f1 = evaluate(retrieved_ids, all_relevant, selected)
        print("\n Evaluation Metrics:")
        print(f"- Precision at 10: {precision:.1%}")
        print(f"- Recall at 100:   {recall:.1%}")
        print(f"- F1 Score:     {f1:.3f}")
        print(f"\n Found {len(all_relevant)} relevant documents")
        print(f"   - In top results: {len(relevant_ids)}")
        print(f"   - Previously missed: {len(false_negatives)}")
    else:
        print("\n No relevant documents identified - cannot calculate metrics")
    
    print("\n" + "="*60)
    print("Thank you for using the Book Search Engine! ")
    print("="*60)