import asyncio
import json
import os
import time
from typing import List
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pypdf import PdfReader

# Load API key from root .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- Pydantic Schemas ---

class CatalystItem(BaseModel):
    company: str
    ticker: str
    sector: str
    industry: str
    catalyst_type: str
    catalyst_topic: str
    catalyst_details: str
    status: str
    observation_date: str # Must be YYYY-MM-DD
    catalyst_dates: str
    exact_sentences_in_the_file: str

class CatalystFocus(BaseModel):
    catalysts: List[CatalystItem]
    file_Id: str

class BullItem(BaseModel):
    company: str
    ticker: str
    sector: str
    industry: str
    observation_date: str # Must be YYYY-MM-DD
    bullish_debate_point: str
    exact_sentences_in_the_file: str

class Evidence(BaseModel):
    evidence_type: str  # "Fact", "Data", "Opinion", "Analyst Projection"
    content: str
    is_objective: bool
    credibility_assessment: str # How this specific piece of evidence strengthens the argument

class SubClaim(BaseModel):
    claim: str
    evidence_list: List[Evidence]

class BullishThesis(BaseModel):
    central_thesis: str
    sub_claims: List[SubClaim]

class BullFocus(BaseModel):
    bullish_views: List[BullItem]
    thesis_analysis: BullishThesis
    file_Id: str

class BearItem(BaseModel):
    company: str
    ticker: str
    sector: str
    industry: str
    observation_date: str # Must be YYYY-MM-DD
    bearish_debate_point: str
    exact_sentences_in_the_file: str

class BearFocus(BaseModel):
    bearish_views: List[BearItem]
    file_Id: str

class FactualItem(BaseModel):
    company: str
    fact_category: str # e.g., "Financials", "Product Specs", "Market Share", "Operational"
    fact_summary: str
    is_recent_update: bool # True if it refers to a recent change or new data point
    original_sentence: str

class FactFocus(BaseModel):
    facts: List[FactualItem]
    file_Id: str

class RelationshipItem(BaseModel):
    source_company: str
    target_company: str
    ticker: str
    relationship_type: str # "Peer", "Competitor", "Supplier", "Customer", "Partner"
    relationship_description: str
    quantifier: str # e.g., "Top 5 customer", "15% of revenue", "Primary foundary partner"
    original_sentence: str

class RelationshipFocus(BaseModel):
    relationships: List[RelationshipItem]
    file_Id: str

# --- Gemini Configuration ---

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash"

def get_pdf_text(file_path):
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

async def extract_catalyst_data(text: str, file_id: str):
    """Async function to extract ALL catalyst items based on a broad definition."""
    start = time.time()
    print(f"   [Agent: CatalystFocus] Extracting ALL catalysts...")
    try:
        # Specific broad definition provided by user
        catalyst_def = (
            "Identify and extract EVERY distinct catalyst. A catalyst is any past or future event or factor "
            "worth tracking that contributes to valuation upside or downside. This includes: "
            "new product/model releases and launches (own or competitor), regulatory changes, "
            "customer order ramps and progress, raw material changes, new technology updates (positive or negative), "
            "soft events (e.g., GTC for semiconductors, spring festival/holidays for consumer companies), "
            "and M&A related activities (mergers, spin-offs, synergies, etc.).\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Observation Date: Return ONLY in YYYY-MM-DD format. If only a month/year is found, use the 1st of the month (e.g., 2022-03-01).\n"
            "2. Meta-data: Identify the Company Name, Ticker, Sector (e.g., Technology), and Industry (e.g., Semiconductors) for EVERY item."
        )
        
        response = await client.aio.models.generate_content(
            model=MODEL_ID,
            contents=f"{catalyst_def}\n\nText: {text}",
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=CatalystFocus,
            ),
        )
        print(f"   [Agent: CatalystFocus] Success! Found items in {time.time()-start:.1f}s")
        data = json.loads(response.text)
        data['file_Id'] = file_id
        return data
    except Exception as e:
        print(f"   [Agent: CatalystFocus] FAILED: {e}")
        return {"catalysts": [], "file_Id": file_id, "error": str(e)}

async def extract_bullish_data(text: str, file_id: str):
    """Async function to extract ALL bullish points and perform structured thesis analysis."""
    start = time.time()
    print(f"   [Agent: BullFocus] Extracting views and building structured thesis...")
    try:
        prompt = (
            "1. Identify EVERY distinct bullish argument or positive driver from the text.\n"
            "2. Identify the 'Central Bullish Thesis' of the document.\n"
            "3. Outline the supporting 'Sub-claims' that build this thesis.\n"
            "4. For each sub-claim, list the supporting evidence (Facts, Data, Opinions).\n"
            "5. Distinguish clearly between objective facts/verifiable data and subjective opinions.\n"
            "6. Assess how each piece of evidence strengthens the argument's credibility.\n"
            "7. Observation Date: Return ONLY in YYYY-MM-DD format.\n"
            "8. Meta-data: Identify Company, Ticker, Sector, and Industry for every item.\n\n"
            f"Text: {text}"
        )
        response = await client.aio.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=BullFocus,
            ),
        )
        print(f"   [Agent: BullFocus] Success! Analyzed thesis in {time.time()-start:.1f}s")
        data = json.loads(response.text)
        data['file_Id'] = file_id
        return data
    except Exception as e:
        print(f"   [Agent: BullFocus] FAILED: {e}")
        return {"bullish_views": [], "thesis_analysis": None, "file_Id": file_id, "error": str(e)}

async def extract_bearish_data(text: str, file_id: str):
    """Async function to extract ALL bearish points."""
    start = time.time()
    print(f"   [Agent: BearFocus] Extracting ALL bearish views...")
    try:
        prompt = (
            "Identify and extract EVERY distinct bearish argument, risk factor, or negative driver from the text below.\n"
            "1. Observation Date: Return ONLY in YYYY-MM-DD format.\n"
            "2. Meta-data: Identify Company, Ticker, Sector, and Industry for every item.\n\n"
            f"Text: {text}"
        )
        response = await client.aio.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=BearFocus,
            ),
        )
        print(f"   [Agent: BearFocus] Success! Found items in {time.time()-start:.1f}s")
        data = json.loads(response.text)
        data['file_Id'] = file_id
        return data
    except Exception as e:
        print(f"   [Agent: BearFocus] FAILED: {e}")
        return {"bearish_views": [], "file_Id": file_id, "error": str(e)}

async def extract_factual_data(text: str, file_id: str):
    """Async function to extract ONLY factual information, filtering out opinions."""
    start = time.time()
    print(f"   [Agent: FactFocus] Extracting Factual Data...")
    try:
        prompt = (
            "Extract EVERY objective fact from the document. A fact is a verifiable data point, historical event, "
            "financial metric, or specific product specification. Exclude all analyst opinions, projections, or 'views'.\n\n"
            "1. Identify the Category (Financials, Product, Market Share, etc.).\n"
            "2. Flag if it is a 'recent update' (happened in the last quarter or announced recently).\n"
            "3. Provide the Key Point Summary and the EXACT Original Sentence.\n\n"
            f"Text: {text}"
        )
        response = await client.aio.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=FactFocus,
            ),
        )
        print(f"   [Agent: FactFocus] Success! Extracted facts in {time.time()-start:.1f}s")
        data = json.loads(response.text)
        data['file_Id'] = file_id
        return data
    except Exception as e:
        print(f"   [Agent: FactFocus] FAILED: {e}")
        return {"facts": [], "file_Id": file_id, "error": str(e)}

async def extract_relationship_data(text: str, file_id: str):
    """Async function to extract company relationships (peers, competitors, supply chain)."""
    start = time.time()
    print(f"   [Agent: RelationshipFocus] Extracting Company Relationships...")
    try:
        prompt = (
            "Identify all companies mentioned in the report and their relationship to the primary company.\n\n"
            "1. Types: Peer (same sector/trends), Competitor (fighting for share), Supplier, Customer, Partner.\n"
            "2. Quantification: Look for specific numbers like 'Top 5 customer', '20% of COGS', etc.\n"
            "3. Original Sentence: Store the sentence that explicitly mentions the relationship.\n"
            "4. Provide Tickers where available.\n\n"
            f"Text: {text}"
        )
        response = await client.aio.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=RelationshipFocus,
            ),
        )
        print(f"   [Agent: RelationshipFocus] Success! Found relationships in {time.time()-start:.1f}s")
        data = json.loads(response.text)
        data['file_Id'] = file_id
        return data
    except Exception as e:
        print(f"   [Agent: RelationshipFocus] FAILED: {e}")
        return {"relationships": [], "file_Id": file_id, "error": str(e)}

# --- Main Orchestrator ---

async def run_extraction(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"\n--- Deep Multi-Agent Analysis: {os.path.basename(file_path)} ---")
    
    print(f"1. Reading PDF...")
    content = get_pdf_text(file_path)
    file_id = os.path.basename(file_path)
    base_filename = os.path.splitext(file_id)[0]
    
    # Trim content to avoid context limits (approx 100k chars)
    content = content[:150000]

    print("2. Launching 5 Agents in Parallel...")
    start_total = time.time()
    
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                extract_catalyst_data(content, file_id),
                extract_bullish_data(content, file_id),
                extract_bearish_data(content, file_id),
                extract_factual_data(content, file_id),
                extract_relationship_data(content, file_id)
            ),
            timeout=400.0
        )
    except asyncio.TimeoutError:
        print(f"\nFATAL ERROR: Timeout limit reached.")
        return
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        return

    combined_json = {
        "catalyst_analysis": results[0],
        "bullish_deep_analysis": results[1],
        "bearish_analysis": results[2],
        "factual_data": results[3],
        "relationship_mapping": results[4]
    }

    # 3. Saving Deep Analysis Output
    print("3. Saving Results to 'output/' folder...")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract thesis for separate file
    bullish_data = results[1].copy()
    thesis_data = bullish_data.pop("thesis_analysis", {})

    file_map = {
        "catalyst_deep": results[0],
        "bullish_views": bullish_data,
        "bullish_thesis": thesis_data,
        "bearish_deep": results[2],
        "factual_data": results[3],
        "relationships": results[4],
        "combined_deep": combined_json
    }

    for suffix, data in file_map.items():
        out_path = os.path.join(output_dir, f"{base_filename}_{suffix}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    print(f"\nFINISHED: Deep Analysis saved to '{output_dir}'")
    print(f"Total time: {time.time() - start_total:.1f}s")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        target_file = r"C:\Users\User\Documents\Financial_Files\BrokerReport\JPM_Broadcom_Inc_Google__2026-01-26_5185915.pdf"
    else:
        target_file = sys.argv[1]

    asyncio.run(run_extraction(target_file))
