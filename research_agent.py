import os
import json
import csv
import asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load API key
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Configuration
INPUT_CSV = "Iran_situation_research_list.csv"
OUTPUT_DIR = "output_research"
MODEL_ID = "gemini-2.5-flash" 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

async def research_company(company_name):
    """Deep dive research for a single company with quantified scoring."""
    print(f"--- Researching {company_name} ---")
    
    prompt = f"""
    Perform a professional financial deep dive for: {company_name}.
    
    RESEARCH TOPICS:
    1. COMPANY PROFILE: Short description, Sector, and Sub-sector.
    2. RETROSPECTIVE (Mar-Jun 2022): 
       - Overview of stock performance.
       - Key market concerns.
       - IMPACT OF FACTORS (XXX): (1) oil, (2) natural gas, (3) shipping, (4) fertilizer, (5) interest rates, (6) stagnation, (7) inflation, (8) asset write-offs, (9) financial stability.
    3. CURRENT GEOPOLITICS (US/Iran Situation):
       - Potential impact, similarities, and differences to Ukraine 2022.
    4. QUANTIFIED SENSITIVITY SCORES (-3 to +3):
       - Assign a score from -3 (Highly Negative) to +3 (Highly Positive) for BOTH the Ukraine 2022 impact AND the potential Iran Situation impact for EACH of these 9 factors:
         [Oil, NatGas, Shipping, Fertilizer, InterestRates, Inflation, Stagnation, AssetRisk, FinancialStability].

    Return the result in this JSON format:
    {{
      "company_name": "{company_name}",
      "company_description": "",
      "sector": "",
      "sub_sector": "",
      "stock_performance_2022": "",
      "market_views_2022": "",
      "factor_impacts": {{
         "revenue": "",
         "cost": "",
         "margin": "",
         "market_share": "",
         "quantified_metrics": ""
      }},
      "ukraine_scores": {{ "oil": 0, "gas": 0, "shipping": 0, "fertilizer": 0, "rates": 0, "inflation": 0, "stagnation": 0, "assets": 0, "financials": 0 }},
      "iran_scores": {{ "oil": 0, "gas": 0, "shipping": 0, "fertilizer": 0, "rates": 0, "inflation": 0, "stagnation": 0, "assets": 0, "financials": 0 }},
      "iran_situation_impact": "",
      "similarities_to_ukraine": "",
      "differences_from_ukraine": "",
      "sector_impact_ukraine_summary": ""
    }}
    """
    
    try:
        response = await client.aio.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        # Handle cases where response text might be wrapped in markdown or is a string
        text = response.text.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(text)
        
        if isinstance(data, str):
            # If it's still a string, it failed to parse as a dict
            print(f"FAILED {company_name}: Response was a string, not a dictionary.")
            return None
            
        safe_name = company_name.replace("/", "_").replace(".", "_").replace(" ", "_")
        out_path = os.path.join(OUTPUT_DIR, f"Research_{safe_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        print(f"Done: {company_name}")
        return data
    except Exception as e:
        print(f"FAILED {company_name}: {e}")
        return None

async def main():
    companies = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                companies.append(line.split(',')[0].strip())
    
    batch_size = 5
    results = []
    for i in range(0, len(companies), batch_size):
        batch = companies[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} of {(len(companies)//batch_size)+1}...")
        batch_results = await asyncio.gather(*[research_company(c) for c in batch])
        results.extend([r for r in batch_results if r])
        await asyncio.sleep(2)
    
    if results:
        csv_path = "Big_Table_Research_Results.csv"
        # Flattened headers for scores
        factors = ["oil", "gas", "shipping", "fertilizer", "rates", "inflation", "stagnation", "assets", "financials"]
        headers = [
            "company_name", "company_description", "sector", "sub_sector", 
            "stock_performance_2022", "market_views_2022", 
            "revenue_impact", "cost_impact", "margin_impact", "market_share_impact", "quantified_metrics",
            "iran_situation_impact", "similarities_to_ukraine", "differences_from_ukraine", "sector_impact_ukraine_summary"
        ]
        headers += [f"UKR_{f.upper()}_SCORE" for f in factors]
        headers += [f"IRAN_{f.upper()}_SCORE" for f in factors]
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in results:
                row = {
                    "company_name": r.get("company_name"),
                    "company_description": r.get("company_description"),
                    "sector": r.get("sector"),
                    "sub_sector": r.get("sub_sector"),
                    "stock_performance_2022": r.get("stock_performance_2022"),
                    "market_views_2022": r.get("market_views_2022"),
                    "revenue_impact": r.get("factor_impacts", {}).get("revenue"),
                    "cost_impact": r.get("factor_impacts", {}).get("cost"),
                    "margin_impact": r.get("factor_impacts", {}).get("margin"),
                    "market_share_impact": r.get("factor_impacts", {}).get("market_share"),
                    "quantified_metrics": r.get("factor_impacts", {}).get("quantified_metrics"),
                    "iran_situation_impact": r.get("iran_situation_impact"),
                    "similarities_to_ukraine": r.get("similarities_to_ukraine"),
                    "differences_from_ukraine": r.get("differences_from_ukraine"),
                    "sector_impact_ukraine_summary": r.get("sector_impact_ukraine_summary")
                }
                # Add Scores
                u_scores = r.get("ukraine_scores", {})
                i_scores = r.get("iran_scores", {})
                for f in factors:
                    row[f"UKR_{f.upper()}_SCORE"] = u_scores.get(f, 0)
                    row[f"IRAN_{f.upper()}_SCORE"] = i_scores.get(f, 0)
                
                writer.writerow(row)
        
        print(f"\n--- ALL DONE ---")
        print(f"Total Companies Researched: {len(results)}")
        print(f"Master CSV Saved: {csv_path}")

if __name__ == "__main__":
    asyncio.run(main())
