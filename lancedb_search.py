import os
import json
import argparse
import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer
from dateutil import parser as date_parser

# --- CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "output")
db_uri = os.path.join(current_dir, "lancedb_data")

# Load embedding model locally
model = SentenceTransformer("all-MiniLM-L6-v2")

def normalize_date(date_str):
    """Attempts to convert any date string to YYYY-MM-DD."""
    if not date_str or date_str == "N/A":
        return "2000-01-01"
    try:
        # date_parser handles "03 June 2025" -> 2025-06-03
        return date_parser.parse(str(date_str)).strftime("%Y-%m-%d")
    except:
        return "2000-01-01"

def get_schema():
    """Defines the Arrow schema for LanceDB."""
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 384)), # all-MiniLM-L6-v2 dim is 384
        pa.field("type", pa.string()),                   # catalyst, bullish, bearish
        pa.field("company", pa.string()),
        pa.field("ticker", pa.string()),
        pa.field("sector", pa.string()),
        pa.field("industry", pa.string()),
        pa.field("observation_date", pa.string()),      # YYYY-MM-DD
        pa.field("topic", pa.string()),
        pa.field("details", pa.string()),
        pa.field("context", pa.string()),                # Added: Full original sentence
        pa.field("file_source", pa.string())
    ])

def index_files():
    """Reads JSON files and loads them into LanceDB."""
    db = lancedb.connect(db_uri)
    table_name = "financial_insights"
    
    # Drop table if exists to refresh (or handle upserts if preferred)
    if table_name in db.table_names():
        db.drop_table(table_name)
    
    table = db.create_table(table_name, schema=get_schema())

    files = [f for f in os.listdir(output_dir) if ("_combined" in f or f.startswith("Research_") or f == "Geopolitics_Comprehensive_Analysis.json") and f.endswith(".json")]
    
    if not files:
        print("No analysis files found to index.")
        return

    all_data = []

    for filename in files:
        file_path = os.path.join(output_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Handle Geopolitics Comprehensive Analysis
                if filename == "Geopolitics_Comprehensive_Analysis.json":
                    # 1. Macro Analysis
                    macro = data.get("macro_analysis", {})
                    # Combine Ukraine drivers/concerns
                    for i, driver in enumerate(macro.get("ukraine_2022_key_drivers", [])):
                        text = f"Ukraine 2022 Driver: {driver}"
                        all_data.append({"id": f"{filename}_ukr_dr_{i}", "vector": model.encode(text).tolist(), "type": "macro_driver", "company": "Macro", "ticker": "N/A", "sector": "N/A", "industry": "N/A", "observation_date": "2022-02-01", "topic": "Ukraine 2022 Driver", "details": driver, "context": driver, "file_source": filename})
                    
                    for i, concern in enumerate(macro.get("ukraine_2022_market_concerns", [])):
                        text = f"Ukraine 2022 Concern: {concern}"
                        all_data.append({"id": f"{filename}_ukr_con_{i}", "vector": model.encode(text).tolist(), "type": "macro_concern", "company": "Macro", "ticker": "N/A", "sector": "N/A", "industry": "N/A", "observation_date": "2022-02-01", "topic": "Ukraine 2022 Concern", "details": concern, "context": concern, "file_source": filename})
                        
                    for i, sec_impact in enumerate(macro.get("ukraine_2022_impact_by_sector", [])):
                        text = f"Ukraine 2022 Impact on {sec_impact.get('sector_name')}: {sec_impact.get('impact_description')}"
                        all_data.append({"id": f"{filename}_ukr_sec_{i}", "vector": model.encode(text).tolist(), "type": "sector_impact", "company": "Sector", "ticker": "N/A", "sector": sec_impact.get('sector_name', 'N/A'), "industry": "N/A", "observation_date": "2022-02-01", "topic": f"Ukraine 2022 Sector Impact", "details": sec_impact.get('impact_description', ''), "context": sec_impact.get('impact_description', ''), "file_source": filename})

                    # Combine Iran drivers/concerns
                    for i, driver in enumerate(macro.get("iran_2026_key_drivers", [])):
                        text = f"Iran 2026 Driver: {driver}"
                        all_data.append({"id": f"{filename}_iran_dr_{i}", "vector": model.encode(text).tolist(), "type": "macro_driver", "company": "Macro", "ticker": "N/A", "sector": "N/A", "industry": "N/A", "observation_date": "2026-01-01", "topic": "Iran 2026 Driver", "details": driver, "context": driver, "file_source": filename})
                    
                    for i, concern in enumerate(macro.get("iran_2026_market_concerns", [])):
                        text = f"Iran 2026 Concern: {concern}"
                        all_data.append({"id": f"{filename}_iran_con_{i}", "vector": model.encode(text).tolist(), "type": "macro_concern", "company": "Macro", "ticker": "N/A", "sector": "N/A", "industry": "N/A", "observation_date": "2026-01-01", "topic": "Iran 2026 Concern", "details": concern, "context": concern, "file_source": filename})
                        
                    for i, sec_impact in enumerate(macro.get("iran_2026_impact_by_sector", [])):
                        text = f"Iran 2026 Impact on {sec_impact.get('sector_name')}: {sec_impact.get('impact_description')}"
                        all_data.append({"id": f"{filename}_iran_sec_{i}", "vector": model.encode(text).tolist(), "type": "sector_impact", "company": "Sector", "ticker": "N/A", "sector": sec_impact.get('sector_name', 'N/A'), "industry": "N/A", "observation_date": "2026-01-01", "topic": f"Iran 2026 Sector Impact", "details": sec_impact.get('impact_description', ''), "context": sec_impact.get('impact_description', ''), "file_source": filename})

                    # 2. Stock Mentions
                    for i, stock in enumerate(data.get("stock_mentions", [])):
                        factors = ", ".join(stock.get("factors_exposed_to", []))
                        text = f"Geopolitical mention of {stock.get('company_name')} ({stock.get('ticker')}). Reason: {stock.get('reason_mentioned')}. Exposed to: {factors}. Original: {stock.get('original_sentences')}"
                        all_data.append({
                            "id": f"{filename}_stock_{i}",
                            "vector": model.encode(text).tolist(),
                            "type": "geopolitics_mention",
                            "company": str(stock.get("company_name", "N/A")),
                            "ticker": str(stock.get("ticker", "N/A")),
                            "sector": "N/A",
                            "industry": "N/A",
                            "observation_date": "N/A",
                            "topic": f"Geopolitics: Exposed to {factors}",
                            "details": str(stock.get("reason_mentioned", "N/A")),
                            "context": str(stock.get("original_sentences", "N/A")),
                            "file_source": filename
                        })
                    continue

                # Handle Research Files
                if filename.startswith("Research_"):
                    topic = data.get("research_metadata", {}).get("topic", "Research")
                    for i, r in enumerate(data.get("findings", [])):
                        # Combine all info for searchability
                        text = f"{r.get('category')} - {r.get('region', r.get('topic', r.get('metric')))}: {r.get('details')} {r.get('context')}"
                        all_data.append({
                            "id": f"{filename}_{i}",
                            "vector": model.encode(text).tolist(),
                            "type": "research",
                            "company": str(r.get("region", r.get("topic", r.get("metric", "N/A")))),
                            "ticker": "N/A",
                            "sector": "N/A",
                            "industry": "N/A",
                            "observation_date": "N/A",
                            "topic": topic,
                            "details": str(r.get("details", "N/A")),
                            "context": str(r.get("context", "N/A")),
                            "file_source": filename
                        })
                    continue

                # Process Catalysts
                for i, c in enumerate(data.get("catalyst_analysis", {}).get("catalysts", [])):
                    text = f"{c.get('catalyst_topic')} {c.get('catalyst_details')}"
                    all_data.append({
                        "id": f"{filename}_cat_{i}",
                        "vector": model.encode(text).tolist(),
                        "type": "catalyst",
                        "company": str(c.get("company", "N/A")),
                        "ticker": str(c.get("ticker", "N/A")),
                        "sector": str(c.get("sector", "N/A")),
                        "industry": str(c.get("industry", "N/A")),
                        "observation_date": normalize_date(c.get("observation_date")),
                        "topic": str(c.get("catalyst_topic", "N/A")),
                        "details": str(c.get("catalyst_details", "N/A")),
                        "context": str(c.get("exact_sentences_in_the_file", "N/A")),
                        "file_source": filename
                    })

                # Process Bullish Views
                for i, b in enumerate(data.get("bullish_deep_analysis", {}).get("bullish_views", [])):
                    text = b.get("bullish_debate_point", "")
                    all_data.append({
                        "id": f"{filename}_bull_{i}",
                        "vector": model.encode(text).tolist(),
                        "type": "bullish",
                        "company": str(b.get("company", "N/A")),
                        "ticker": str(b.get("ticker", "N/A")),
                        "sector": str(b.get("sector", "N/A")),
                        "industry": str(b.get("industry", "N/A")),
                        "observation_date": normalize_date(b.get("observation_date")),
                        "topic": "Bullish Debate Point",
                        "details": text,
                        "context": str(b.get("exact_sentences_in_the_file", "N/A")),
                        "file_source": filename
                    })

                # Process Bearish Views
                for i, br in enumerate(data.get("bearish_analysis", {}).get("bearish_views", [])):
                    text = br.get("bearish_debate_point", "")
                    all_data.append({
                        "id": f"{filename}_bear_{i}",
                        "vector": model.encode(text).tolist(),
                        "type": "bearish",
                        "company": str(br.get("company", "N/A")),
                        "ticker": str(br.get("ticker", "N/A")),
                        "sector": str(br.get("sector", "N/A")),
                        "industry": str(br.get("industry", "N/A")),
                        "observation_date": normalize_date(br.get("observation_date")),
                        "topic": "Bearish Debate Point",
                        "details": text,
                        "context": str(br.get("exact_sentences_in_the_file", "N/A")),
                        "file_source": filename
                    })

                # Process Factual Data
                for i, f_item in enumerate(data.get("factual_data", {}).get("facts", [])):
                    text = f_item.get("fact_summary", "")
                    all_data.append({
                        "id": f"{filename}_fact_{i}",
                        "vector": model.encode(text).tolist(),
                        "type": "fact",
                        "company": str(f_item.get("company", "N/A")),
                        "ticker": "N/A",
                        "sector": "N/A",
                        "industry": "N/A",
                        "observation_date": "N/A", # Facts are often general
                        "topic": str(f_item.get("fact_category", "Fact")),
                        "details": text,
                        "context": str(f_item.get("original_sentence", "N/A")),
                        "file_source": filename
                    })

                # Process Relationships
                for i, rel in enumerate(data.get("relationship_mapping", {}).get("relationships", [])):
                    text = f"{rel.get('source_company')} {rel.get('relationship_type')} {rel.get('target_company')}: {rel.get('relationship_description')}"
                    all_data.append({
                        "id": f"{filename}_rel_{i}",
                        "vector": model.encode(text).tolist(),
                        "type": "relationship",
                        "company": str(rel.get("source_company", "N/A")),
                        "ticker": str(rel.get("ticker", "N/A")),
                        "sector": "N/A",
                        "industry": "N/A",
                        "observation_date": "N/A",
                        "topic": f"{rel.get('relationship_type')} - {rel.get('target_company')}",
                        "details": text,
                        "context": str(rel.get("original_sentence", "N/A")),
                        "file_source": filename
                    })
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if all_data:
        table.add(all_data)
        print(f"Successfully indexed {len(all_data)} items into LanceDB.")
    else:
        print("No data to index.")

def search(query=None, company=None, sector=None, start_date=None, end_date=None, limit=10):
    """Performs a hybrid SQL + Vector search."""
    db = lancedb.connect(db_uri)
    if "financial_insights" not in db.table_names():
        print("Table not found. Please run with --index first.")
        return
    
    table = db.open_table("financial_insights")
    
    # Build SQL Filter
    filters = []
    if company:
        filters.append(f"company = '{company}'")
    if sector:
        filters.append(f"sector = '{sector}'")
    if start_date:
        filters.append(f"observation_date >= '{start_date}'")
    if end_date:
        filters.append(f"observation_date <= '{end_date}'")
    
    where_clause = " AND ".join(filters) if filters else None

    print(f"Searching for: '{query}'")
    if where_clause:
        print(f"Applying Filters: {where_clause}")

    # Hybrid Search Execution
    if query:
        query_vector = model.encode(query).tolist()
        results = table.search(query_vector).where(where_clause).limit(limit).to_pandas()
    else:
        # Just SQL filtering if no semantic query
        results = table.to_pandas()
        if where_clause:
            results = table.search().where(where_clause).limit(limit).to_pandas()

    if results.empty:
        print("No matches found.")
        return

    print("\n--- LanceDB Hybrid Search Results ---")
    for _, row in results.iterrows():
        score_str = f" [Score: {1-row.get('_distance', 0):.2f}]" if '_distance' in row else ""
        print(f"\n{row['company']} ({row['ticker']}) - {row['observation_date']}{score_str}")
        print(f"TYPE:    {row['type'].upper()}")
        print(f"TOPIC:   {row['topic']}")
        print(f"DETAILS: {row['details'][:500]}")
        print(f"CONTEXT: {row.get('context', 'N/A')}")
        print(f"SOURCE:  {row['file_source']}")
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LanceDB Financial Search")
    parser.add_argument("--index", action="store_true", help="Index files in output/ to LanceDB")
    parser.add_argument("--query", help="Semantic search query")
    parser.add_argument("--company", help="Filter by company")
    parser.add_argument("--sector", help="Filter by sector")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=5, help="Result limit")

    args = parser.parse_args()

    if args.index:
        index_files()
    
    if args.query or args.company or args.sector or args.start or args.end:
        search(
            query=args.query,
            company=args.company,
            sector=args.sector,
            start_date=args.start,
            end_date=args.end,
            limit=args.limit
        )
    elif not args.index:
        print("Usage: python lancedb_search.py --query 'inflation' --start '2022-03-01' --end '2022-05-31'")
