import os
import json
import csv

# --- CONFIGURATION ---
OUTPUT_DIR = "output_research"
CSV_PATH = "Big_Table_Research_Results_v2.csv"

def update_csv():
    results = []
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("Research_") and f.endswith(".json")]
    
    if not files:
        print("No research files found.")
        return

    print(f"Processing {len(files)} files to recalculate Big Table...")

    for filename in files:
        with open(os.path.join(OUTPUT_DIR, filename), "r", encoding="utf-8") as f:
            r = json.load(f)
            
            # Calculate Averages
            factors = ["oil", "gas", "shipping", "fertilizer", "rates", "inflation", "stagnation", "assets", "financials"]
            
            u_scores = r.get("ukraine_scores", {})
            i_scores = r.get("iran_scores", {})
            
            u_vals = [u_scores.get(f, 0) for f in factors]
            i_vals = [i_scores.get(f, 0) for f in factors]
            
            r["ukr_avg"] = sum(u_vals) / len(u_vals) if u_vals else 0
            r["iran_avg"] = sum(i_vals) / len(i_vals) if i_vals else 0
            
            results.append(r)

    # Export Big Table (CSV)
    factors_upper = ["OIL", "GAS", "SHIPPING", "FERTILIZER", "RATES", "INFLATION", "STAGNATION", "ASSETS", "FINANCIALS"]
    headers = [
        "company_name", "company_description", "sector", "sub_sector", 
        "stock_performance_2022", "market_views_2022", 
        "revenue_impact", "cost_impact", "margin_impact", "market_share_impact", "quantified_metrics",
        "iran_situation_impact", "similarities_to_ukraine", "differences_from_ukraine", "sector_impact_ukraine_summary",
        "UKR_AVG_SCORE", "IRAN_AVG_SCORE"
    ]
    headers += [f"UKR_{f}_SCORE" for f in factors_upper]
    headers += [f"IRAN_{f}_SCORE" for f in factors_upper]
    
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
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
                "sector_impact_ukraine_summary": r.get("sector_impact_ukraine_summary"),
                "UKR_AVG_SCORE": round(r.get("ukr_avg", 0), 2),
                "IRAN_AVG_SCORE": round(r.get("iran_avg", 0), 2)
            }
            
            u_scores = r.get("ukraine_scores", {})
            i_scores = r.get("iran_scores", {})
            for f in factors:
                row[f"UKR_{f.upper()}_SCORE"] = u_scores.get(f, 0)
                row[f"IRAN_{f.upper()}_SCORE"] = i_scores.get(f, 0)
            
            writer.writerow(row)
    
    print(f"Success: {CSV_PATH} updated with average scores.")

if __name__ == "__main__":
    update_csv()
