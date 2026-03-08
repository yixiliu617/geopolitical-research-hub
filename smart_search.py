import os
import json
import asyncio
import argparse
from google import genai
from dotenv import load_dotenv

# Load API key from root .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

async def perform_smart_search(company_name, query):
    output_dir = "output"
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' not found.")
        return

    # 1. PRE-FILTERING (Local Python Efficiency)
    # We only look at combined files
    files = [f for f in os.listdir(output_dir) if "_combined" in f and f.endswith(".json")]
    
    if not files:
        print("No analysis files found.")
        return

    # Extract only the meaningful text fields to save tokens and speed up the AI
    search_corpus = []
    for filename in files:
        try:
            with open(os.path.join(output_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Check if this file is actually about the requested company
                file_text = json.dumps(data).lower()
                if company_name.lower() not in file_text:
                    continue

                # Extract only the "meat" of the data
                snippets = []
                
                # Catalysts
                for c in data.get("catalyst_analysis", {}).get("catalysts", []):
                    snippets.append(f"[Catalyst] {c.get('catalyst_topic')}: {c.get('catalyst_details')}")
                
                # Bullish Views
                for b in data.get("bullish_deep_analysis", {}).get("bullish_views", []):
                    snippets.append(f"[Bullish] {b.get('bullish_debate_point')}")
                
                # Bearish Views
                for br in data.get("bearish_analysis", {}).get("bearish_views", []):
                    snippets.append(f"[Bearish] {br.get('bearish_debate_point')}")

                if snippets:
                    search_corpus.append({
                        "file": filename,
                        "content": "\n".join(snippets)
                    })
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

    if not search_corpus:
        print(f"No mentions of '{company_name}' found in your files.")
        return

    print(f"Found {len(search_corpus)} relevant files. Asking Gemini for semantic matches for '{query}'...")

    # 2. SEMANTIC SEARCH (AI Intelligence)
    prompt = f"""
    You are a financial search expert. 
    QUERY: '{query}' (Include synonyms like 'Release', 'Ramp', 'Deployment', etc.)
    COMPANY: '{company_name}'

    From the data below, identify the specific items that match the query. 
    For each match, provide:
    - The File Name
    - A brief summary of the finding
    - The type (Catalyst/Bullish/Bearish)

    DATA:
    {json.dumps(search_corpus)}
    """

    try:
        # Using gemini-2.5-flash (Working ID)
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        print("\n--- Efficient Smart Search Results ---")
        print(response.text)
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficient smart search.")
    parser.add_argument("--company", required=True, help="Company name (e.g. Broadcom)")
    parser.add_argument("--query", required=True, help="Search query (e.g. Launch)")
    
    args = parser.parse_args()
    asyncio.run(perform_smart_search(args.company, args.query))
