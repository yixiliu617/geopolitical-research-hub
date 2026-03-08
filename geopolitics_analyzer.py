import os
import json
import asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pypdf import PdfReader
from pydantic import BaseModel
from typing import List, Optional

# Load API key
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash"

class SectorImpact(BaseModel):
    sector_name: str
    impact_description: str
    quantified_score: float # Score from -3 (Very Negative) to +3 (Very Positive)

class MacroAnalysis(BaseModel):
    ukraine_2022_key_drivers: List[str]
    ukraine_2022_market_concerns: List[str]
    ukraine_2022_impact_by_sector: List[SectorImpact]
    iran_2026_key_drivers: List[str]
    iran_2026_market_concerns: List[str]
    iran_2026_impact_by_sector: List[SectorImpact]

class StockMention(BaseModel):
    company_name: str
    ticker: Optional[str]
    mentioned_date: str 
    reason_mentioned: str
    factors_exposed_to: List[str]
    original_sentences: str
    source_file: str

class GeopoliticsReport(BaseModel):
    macro_analysis: MacroAnalysis
    stock_mentions: List[StockMention]

def get_pdf_text(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted: text += extracted + "\n"
        return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

async def analyze_single_pdf(file_path):
    filename = os.path.basename(file_path)
    print(f"   Analysing: {filename}...")
    text = get_pdf_text(file_path)
    
    prompt = """
    Perform a geopolitical and financial analysis of the provided report.
    
    1. Extract Macro drivers, concerns, and sector impacts for Ukraine 2022 and/or Iran 2026.
       - IMPORTANT: Group all findings into these STANDARD SECTORS: 
         [Technology, Energy, Financials, Industrials, Consumer Discretionary, Consumer Staples, Healthcare, Utilities, Materials, Real Estate, Communication Services].
       - For each standard sector, provide a 'impact_description' summarizing all relevant points found in the text.
       - Provide a 'quantified_score' from -3 (Highly Negative) to +3 (Highly Positive) for the sector.
    
    2. Record ALL single stocks mentioned. For each:
       - Name, Ticker, Date (YYYY-MM-DD), Reason, Exposure Factors, Original Sentence.
    
    Return JSON.
    """
    
    try:
        response = await client.aio.models.generate_content(
            model=MODEL_ID,
            contents=f"{prompt}\n\nTEXT:\n{text}",
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=GeopoliticsReport,
            ),
        )
        res_text = response.text.strip()
        if res_text.startswith("```json"):
            res_text = res_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(res_text)
        
        # Ensure source_file is set correctly for all mentions
        for s in data.get("stock_mentions", []):
            s["source_file"] = filename
            
        return data
    except Exception as e:
        print(f"   FAILED {filename}: {e}")
        return None

async def run_analysis():
    folder_path = r"C:\Users\User\Documents\Financial_Files\Geopolitical_sit"
    output_dir = "output"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    
    master_macro = {
        "ukraine_2022_key_drivers": [], "ukraine_2022_market_concerns": [], "ukraine_2022_impact_by_sector": [],
        "iran_2026_key_drivers": [], "iran_2026_market_concerns": [], "iran_2026_impact_by_sector": []
    }
    master_stock_mentions = []

    print(f"Processing {len(files)} files sequentially...")
    
    for f in files:
        data = await analyze_single_pdf(f)
        if data:
            # Merge Macro
            m = data.get("macro_analysis", {})
            master_macro["ukraine_2022_key_drivers"].extend(m.get("ukraine_2022_key_drivers", []))
            master_macro["ukraine_2022_market_concerns"].extend(m.get("ukraine_2022_market_concerns", []))
            master_macro["ukraine_2022_impact_by_sector"].extend(m.get("ukraine_2022_impact_by_sector", []))
            master_macro["iran_2026_key_drivers"].extend(m.get("iran_2026_key_drivers", []))
            master_macro["iran_2026_market_concerns"].extend(m.get("iran_2026_market_concerns", []))
            master_macro["iran_2026_impact_by_sector"].extend(m.get("iran_2026_impact_by_sector", []))
            
            # Merge Stocks
            master_stock_mentions.extend(data.get("stock_mentions", []))

    # Deduplicate lists
    for k in ["ukraine_2022_key_drivers", "ukraine_2022_market_concerns", "iran_2026_key_drivers", "iran_2026_market_concerns"]:
        master_macro[k] = list(set(master_macro[k]))

    final_report = {
        "macro_analysis": master_macro,
        "stock_mentions": master_stock_mentions
    }

    out_path = os.path.join(output_dir, "Geopolitics_Comprehensive_Analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
        
    print(f"\nSUCCESS! Aggregated analysis saved to {out_path}")
    print(f"Total Stock Mentions: {len(master_stock_mentions)}")

if __name__ == "__main__":
    asyncio.run(run_analysis())
