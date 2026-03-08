import os
import json
import argparse
import chromadb
from chromadb.utils import embedding_functions

# 1. SETUP LOCAL DIRECTORIES
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "output")
db_dir = os.path.join(current_dir, "local_db")

client = chromadb.PersistentClient(path=db_dir)
default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def index_files():
    """Reads JSON files and loads them into the local vector database with full metadata."""
    collection = client.get_or_create_collection(name="financial_insights_v2", embedding_function=default_ef)
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory {output_dir} not found.")
        return

    files = [f for f in os.listdir(output_dir) if "_combined" in f and f.endswith(".json")]
    
    if not files:
        print("No analysis files found to index.")
        return

    print(f"Indexing {len(files)} files into local vector DB...")
    
    ids = []
    documents = []
    metadatas = []

    for filename in files:
        file_path = os.path.join(output_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Extract Catalysts with FULL metadata
                for i, c in enumerate(data.get("catalyst_analysis", {}).get("catalysts", [])):
                    # The searchable content
                    text = f"{c.get('catalyst_topic')} {c.get('catalyst_details')}"
                    documents.append(text)
                    ids.append(f"{filename}_cat_{i}")
                    
                    # Store everything in metadata
                    metadatas.append({
                        "file": filename,
                        "company": str(c.get("company", "N/A")),
                        "type": "catalyst",
                        "catalyst_type": str(c.get("catalyst_type", "N/A")),
                        "catalyst_topic": str(c.get("catalyst_topic", "N/A")),
                        "catalyst_details": str(c.get("catalyst_details", "N/A")),
                        "status": str(c.get("status", "N/A")),
                        "observation_date": str(c.get("observation_date", "N/A")),
                        "catalyst_dates": str(c.get("catalyst_dates", "N/A"))
                    })

                # Bullish Views (Simplified metadata as they use different schema)
                for i, b in enumerate(data.get("bullish_deep_analysis", {}).get("bullish_views", [])):
                    documents.append(b.get("bullish_debate_point", ""))
                    ids.append(f"{filename}_bull_{i}")
                    metadatas.append({
                        "file": filename,
                        "company": str(b.get("company", "N/A")),
                        "type": "bullish",
                        "catalyst_topic": "Bullish Debate Point",
                        "catalyst_details": str(b.get("bullish_debate_point", "N/A")),
                        "observation_date": str(b.get("observation_date", "N/A")),
                        "status": "Positive Outlook"
                    })

                # Bearish Views
                for i, br in enumerate(data.get("bearish_analysis", {}).get("bearish_views", [])):
                    documents.append(br.get("bearish_debate_point", ""))
                    ids.append(f"{filename}_bear_{i}")
                    metadatas.append({
                        "file": filename,
                        "company": str(br.get("company", "N/A")),
                        "type": "bearish",
                        "catalyst_topic": "Bearish Debate Point",
                        "catalyst_details": str(br.get("bearish_debate_point", "N/A")),
                        "observation_date": str(br.get("observation_date", "N/A")),
                        "status": "Negative Risk"
                    })
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if documents:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"Successfully indexed {len(documents)} snippets locally.")
    else:
        print("No meaningful data found inside the JSON files.")

def search(company, query):
    """Performs semantic search and displays full metadata."""
    collection = client.get_collection(name="financial_insights_v2", embedding_function=default_ef)
    
    print(f"Searching local DB for '{query}' regarding '{company}'...")
    
    # Building the filter
    where_filter = {"company": company} if company else None

    results = collection.query(
        query_texts=[query],
        n_results=5,
        where=where_filter
    )

    print("\n--- Detailed Local RAG Search Results ---")
    
    if not results['documents'] or not results['documents'][0]:
        print("No matches found.")
        return

    for i in range(len(results['documents'][0])):
        meta = results['metadatas'][0][i]
        dist = results['distances'][0][i]
        score = 1 - dist
        
        print(f"\n[{i+1}] SCORE: {score:.2f}")
        print(f"    FILE:             {meta.get('file')}")
        print(f"    COMPANY:          {meta.get('company')}")
        print(f"    TYPE:             {meta.get('type')}")
        print(f"    CATALYST TOPIC:   {meta.get('catalyst_topic')}")
        print(f"    CATALYST TYPE:    {meta.get('catalyst_type', 'N/A')}")
        print(f"    DETAILS:          {meta.get('catalyst_details')[:200]}...")
        print(f"    STATUS:           {meta.get('status')}")
        print(f"    OBSERVATION DATE: {meta.get('observation_date')}")
        print(f"    CATALYST DATE:    {meta.get('catalyst_dates', 'N/A')}")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Detailed RAG Search")
    parser.add_argument("--index", action="store_true", help="Refresh the local index first")
    parser.add_argument("--company", help="Filter by company name")
    parser.add_argument("--query", help="Semantic search query")

    args = parser.parse_args()

    if args.index:
        index_files()
    
    if args.query:
        search(args.company, args.query)
    elif not args.index:
        print("Usage: python local_rag_search.py --query 'Your Query' [--company 'Broadcom Inc'] [--index]")
