import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import re
import lancedb
from google import genai
from google.genai import types
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
CSV_PATH = "Big_Table_Research_Results_v2.csv"
RESEARCH_DIR = "output_research"
GEOPOL_JSON = "output/Geopolitics_Comprehensive_Analysis.json"
DB_URI = "lancedb_data"

st.set_page_config(page_title="Geopolitical Research Hub", layout="wide")

# Load Models
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash"

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    return df

@st.cache_data
def load_geopol_report():
    if os.path.exists(GEOPOL_JSON):
        with open(GEOPOL_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # AGGREGATION LOGIC: Group multiple mentions of the same sector
            def aggregate_sectors(impact_list):
                if not impact_list: return []
                agg = {}
                for item in impact_list:
                    name = item['sector_name']
                    if name not in agg:
                        agg[name] = {"scores": [], "desc": []}
                    agg[name]["scores"].append(item['quantified_score'])
                    agg[name]["desc"].append(item['impact_description'])
                
                final = []
                for name, vals in agg.items():
                    avg_score = round(sum(vals["scores"]) / len(vals["scores"]), 2)
                    # Join descriptions and remove duplicates/shorten
                    unique_descs = list(set(vals["desc"]))
                    combined_desc = " ".join(unique_descs[:3]) # Limit to top 3 snippets for brevity
                    final.append({
                        "sector_name": name,
                        "quantified_score": avg_score,
                        "impact_description": combined_desc
                    })
                return sorted(final, key=lambda x: x['sector_name'])

            # Update the macro analysis with aggregated data
            if 'macro_analysis' in data:
                m = data['macro_analysis']
                m['ukraine_2022_impact_by_sector'] = aggregate_sectors(m.get('ukraine_2022_impact_by_sector', []))
                m['iran_2026_impact_by_sector'] = aggregate_sectors(m.get('iran_2026_impact_by_sector', []))
            
            return data
    return None

def perform_semantic_search(query, limit=5):
    db = lancedb.connect(DB_URI)
    if "financial_insights" not in db.list_tables():
        return pd.DataFrame()
    table = db.open_table("financial_insights")
    query_vector = embedding_model.encode(query).tolist()
    results = table.search(query_vector).limit(limit).to_pandas()
    return results

def get_company_details(company_name):
    ticker_match = re.search(r'\((.*?)\)', company_name)
    ticker = ticker_match.group(1) if ticker_match else company_name
    def normalize(s):
        s = s.replace("/", "_").replace(".", "_").replace(" ", "_")
        return re.sub(r'_+', '_', s).strip("_").lower()
    target_norm = normalize(ticker)
    if not os.path.exists(RESEARCH_DIR): return None
    all_files = os.listdir(RESEARCH_DIR)
    for f in all_files:
        if f.startswith("Research_") and f.endswith(".json"):
            file_norm = normalize(f.replace("Research_", "").replace(".json", ""))
            if target_norm == file_norm or target_norm in file_norm or file_norm in target_norm:
                with open(os.path.join(RESEARCH_DIR, f), "r", encoding="utf-8") as file:
                    return json.load(file)
    return None

# --- SIDEBAR FILTERS (LEFT) ---
st.sidebar.title("📊 Global Filters")
df = load_data()
geopol_data = load_geopol_report()

all_sectors = sorted(df['sector'].dropna().unique().tolist())
selected_sector = st.sidebar.multiselect("Filter by Sector", all_sectors)

st.sidebar.subheader("Sensitivity Screening")
score_type = st.sidebar.selectbox("Screening Scenario", ["UKR", "IRAN"])
factors_list_screen = ["AVG", "OIL", "GAS", "SHIPPING", "FERTILIZER", "RATES", "INFLATION", "STAGNATION", "ASSETS", "FINANCIALS"]
factor_to_filter = st.sidebar.selectbox("Screening Factor", factors_list_screen)
min_score, max_score = st.sidebar.slider("Score Range", -3.0, 3.0, (-3.0, 3.0), step=0.1)

search_query = st.sidebar.text_input("🔍 Search Text Filter")

# --- DATA FILTERING ---
filtered_df = df.copy()
if selected_sector:
    filtered_df = filtered_df[filtered_df['sector'].isin(selected_sector)]

score_col_filter = f"{score_type}_{factor_to_filter}_SCORE" if factor_to_filter != "AVG" else f"{score_type}_AVG_SCORE"
if score_col_filter in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df[score_col_filter] >= min_score) & (filtered_df[score_col_filter] <= max_score)]

if search_query:
    filtered_df = filtered_df[filtered_df['company_name'].str.contains(search_query, case=False) | filtered_df['company_description'].str.contains(search_query, case=False)]

# --- LAYOUT SPLIT (MAIN vs CHAT) ---
main_col, chat_col = st.columns([3, 1.2])

with main_col:
    st.title("🌎 Geopolitical Research Hub")
    
    tab_dashboard, tab_report, tab_search = st.tabs(["📊 Sensitivity Dashboard", "🌍 Macro Analysis Report", "🔍 AI Semantic Search"])

    with tab_dashboard:
        # 1. SECTOR SENSITIVITY SUMMARY
        st.subheader("🏢 Sector Sensitivity Summary")
        score_cols = [c for c in df.columns if "SCORE" in c]
        sector_summary = filtered_df.groupby("sector")[score_cols].mean().round(2)
        
        short_col_map = {c: c.replace("_SCORE", "").replace("UKR_", "🇺🇦 ").replace("IRAN_", "🇮🇷 ") for c in score_cols}
        sector_summary_renamed = sector_summary.rename(columns=short_col_map)

        def color_scores_simple(val):
            if not isinstance(val, (int, float)): return ""
            if val > 0:
                alpha = min(1.0, abs(val)/3); return f'background-color: rgba(0, 128, 0, {alpha}); color: black'
            elif val < 0:
                alpha = min(1.0, abs(val)/3); return f'background-color: rgba(255, 0, 0, {alpha}); color: black'
            return 'background-color: white; color: black'

        col_config = {col: st.column_config.NumberColumn(col, width="small") for col in sector_summary_renamed.columns}
        st.dataframe(sector_summary_renamed.style.applymap(color_scores_simple), use_container_width=True, column_config=col_config)

        # 2. SINGLE STOCK HEATMAP
        st.divider()
        st.subheader("🔥 Company Sensitivity Heatmap")
        
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: viz_mode = st.radio("Focus", ["Iran Situation", "Ukraine 2022", "Side-by-Side"], horizontal=True)
        with c2: 
            factors_list = ["OIL", "GAS", "SHIPPING", "FERTILIZER", "RATES", "INFLATION", "STAGNATION", "ASSETS", "FINANCIALS"]
            sort_options = ["company_name", "UKR_AVG_SCORE", "IRAN_AVG_SCORE"] + [f"UKR_{f}_SCORE" for f in factors_list] + [f"IRAN_{f}_SCORE" for f in factors_list]
            sort_col = st.selectbox("Sort By", sort_options, index=2)
        with c3: sort_order = st.radio("Order", ["Desc", "Asc"], horizontal=True)

        filtered_df = filtered_df.sort_values(by=sort_col, ascending=(sort_order == "Asc"))
        
        if not filtered_df.empty:
            ukr_cols = ["UKR_AVG_SCORE"] + [f"UKR_{f}_SCORE" for f in factors_list]
            iran_cols = ["IRAN_AVG_SCORE"] + [f"IRAN_{f}_SCORE" for f in factors_list]
            heatmap_data = filtered_df.set_index("company_name")
            
            if viz_mode == "Iran Situation":
                display_cols = iran_cols
                labels = [c.replace("_SCORE", "").replace("IRAN_", "") for c in display_cols]
            elif viz_mode == "Ukraine 2022":
                display_cols = ukr_cols
                labels = [c.replace("_SCORE", "").replace("UKR_", "") for c in display_cols]
            else:
                display_cols = ["IRAN_AVG_SCORE", "UKR_AVG_SCORE"]
                for f in factors_list: display_cols.extend([f"IRAN_{f}_SCORE", f"UKR_{f}_SCORE"])
                labels = [c.replace("_SCORE", "").replace("IRAN_", "🇮🇷 ").replace("UKR_", "🇺🇦 ") for c in display_cols]

            fig = px.imshow(heatmap_data[display_cols],
                            labels=dict(x="Factor", y="Company", color="Score"),
                            x=labels, y=heatmap_data.index,
                            color_continuous_scale=[[0, 'red'], [0.5, 'white'], [1, 'green']],
                            range_color=[-3, 3], aspect="auto", height=max(400, len(filtered_df) * 22))
            fig.update_layout(xaxis={'side': 'top'}, xaxis_nticks=len(display_cols))
            st.plotly_chart(fig, use_container_width=True)

        # 3. DETAILED INSIGHTS GRID
        st.divider()
        st.subheader("📋 Detailed Insights Grid")
        def get_score_color(val):
            if not isinstance(val, (int, float)): return "#ffffff"
            if val > 0: alpha = min(1.0, abs(val)/3); return f'rgba(0, 128, 0, {alpha})'
            elif val < 0: alpha = min(1.0, abs(val)/3); return f'rgba(255, 0, 0, {alpha})'
            return "#ffffff"

        html_table = """<style>
            .scroll-container { height: 600px; overflow-y: auto; border: 1px solid #ddd; border-radius: 5px; }
            .report-table { width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 13px; }
            .report-table th { background-color: #f0f2f6; position: sticky; top: 0; z-index: 10; padding: 12px; border: 1px solid #ddd; text-align: left; }
            .report-table td { padding: 12px; border: 1px solid #ddd; vertical-align: top; line-height: 1.6; background-color: white; }
            .company-name { font-weight: bold; color: #1f77b4; min-width: 140px; }
            .score-cell { text-align: center; font-weight: bold; width: 40px; border: 1px solid #eee; }
            .text-cell { min-width: 300px; white-space: normal; word-wrap: break-word; }
        </style><div class="scroll-container"><table class="report-table"><thead><tr><th>Company</th><th>Sector</th><th>2022 Performance</th><th>🇮🇷 Iran Analysis</th><th>Similarities</th><th>Differences</th>""" + "".join([f"<th class='score-cell'>{c.replace('_SCORE', '').replace('IRAN_', '🇮🇷').replace('UKR_', '🇺🇦')}</th>" for c in score_cols]) + "</tr></thead><tbody>"

        for _, row in filtered_df.iterrows():
            html_table += f"<tr><td class='company-name'>{row['company_name']}</td><td>{row['sector']}</td><td class='text-cell'>{row['stock_performance_2022']}</td><td class='text-cell'>{row['iran_situation_impact']}</td><td class='text-cell'>{row['similarities_to_ukraine']}</td><td class='text-cell'>{row['differences_from_ukraine']}</td>"
            for sc in score_cols:
                html_table += f"<td class='score-cell' style='background-color: {get_score_color(row[sc])};'>{row[sc]}</td>"
            html_table += "</tr>"
        st.markdown(html_table + "</tbody></table></div>", unsafe_allow_html=True)

        # 4. COMPANY DEEP DIVE (Dashboard Tab Only)
        st.divider()
        st.subheader("🔍 Company Deep Dive")
        selected_company = st.selectbox("Select a company to load full research report", filtered_df['company_name'].tolist())
        if selected_company:
            details = get_company_details(selected_company)
            if details:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.info(f"**{details['company_name']}**\n\n{details['company_description']}")
                    st.metric("Sector", details['sector']); st.metric("Sub-Sector", details['sub_sector'])
                    st.write("### Ukraine 2022 Scores"); st.json(details['ukraine_scores'])
                    st.write("### Iran Situation Scores"); st.json(details['iran_scores'])
                with col2:
                    tab1, tab2, tab3 = st.tabs(["Iran Impact Analysis", "2022 Retrospective", "Sector Context"])
                    with tab1:
                        st.write("#### 🇮🇷 Potential US / Iran Impact"); st.write(details['iran_situation_impact'])
                        st.write("**Similarities to Ukraine:**"); st.write(details['similarities_to_ukraine'])
                        st.write("**Differences from Ukraine:**"); st.write(details['differences_from_ukraine'])
                    with tab2:
                        st.write("#### ⏪ Mar-Jun 2022 Analysis"); st.write(f"**Performance:** {details['stock_performance_2022']}")
                        st.write(f"**Market Views:** {details['market_views_2022']}"); st.table(details['factor_impacts'])
                    with tab3:
                        st.write("#### 🏭 Overall Sector Impact (Ukraine)"); st.write(details['sector_impact_ukraine_summary'])

    with tab_report:
        if geopol_data:
            st.subheader("🚀 Comprehensive Geopolitical Analysis")
            m = geopol_data['macro_analysis']
            
            # Formatting fix: Use custom CSS for narrow columns and forced wrapping
            st.markdown("""
                <style>
                .macro-col { padding: 15px; border: 1px solid #e6e9ef; border-radius: 10px; background-color: #f9fbfd; min-height: 350px; word-wrap: break-word; overflow-wrap: break-word; }
                .macro-title { font-weight: bold; font-size: 18px; margin-bottom: 10px; }
                .macro-list { font-size: 13px; line-height: 1.4; }
                </style>
            """, unsafe_allow_html=True)

            def summarize_list(items, limit=8):
                if not items: return "N/A"
                summary = items[:limit]
                return ", ".join(summary) + ("..." if len(items) > limit else "")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="macro-col"><div class="macro-title">🇺🇦 Ukraine 2022 Summary</div><div class="macro-list"><b>Key Drivers:</b> {summarize_list(m["ukraine_2022_key_drivers"])}<br><br><b>Market Concerns:</b> {summarize_list(m["ukraine_2022_market_concerns"])}</div></div>', unsafe_allow_html=True)
                with st.expander("Show Detailed Sector Impacts (Ukraine)"):
                    for s in m['ukraine_2022_impact_by_sector']: 
                        st.markdown(f"**{s['sector_name']}** ({s['quantified_score']}): {s['impact_description']}")
            with c2:
                st.markdown(f'<div class="macro-col"><div class="macro-title">🇮🇷 Iran 2026 Summary</div><div class="macro-list"><b>Key Drivers:</b> {summarize_list(m["iran_2026_key_drivers"])}<br><br><b>Market Concerns:</b> {summarize_list(m["iran_2026_market_concerns"])}</div></div>', unsafe_allow_html=True)
                with st.expander("Show Detailed Sector Impacts (Iran)"):
                    for s in m['iran_2026_impact_by_sector']: 
                        st.markdown(f"**{s['sector_name']}** ({s['quantified_score']}): {s['impact_description']}")

            # --- MACRO SECTOR HEATMAP ---
            st.divider()
            st.subheader("🔥 Macro View: Sector Sensitivity Heatmap")
            st.markdown("This heatmap represents the **theoretical sector impact** as described in the geopolitical reports.")
            
            # Prepare Macro Sector Data
            ukr_sec_data = {s['sector_name']: s['quantified_score'] for s in m['ukraine_2022_impact_by_sector']}
            iran_sec_data = {s['sector_name']: s['quantified_score'] for s in m['iran_2026_impact_by_sector']}
            
            all_sec_names = sorted(list(set(list(ukr_sec_data.keys()) + list(iran_sec_data.keys()))))
            macro_heatmap_df = pd.DataFrame(index=all_sec_names)
            macro_heatmap_df['🇺🇦 Ukraine 2022'] = macro_heatmap_df.index.map(ukr_sec_data).fillna(0)
            macro_heatmap_df['🇮🇷 Iran 2026'] = macro_heatmap_df.index.map(iran_sec_data).fillna(0)
            
            fig_macro = px.imshow(macro_heatmap_df.T,
                                labels=dict(x="Sector", y="Scenario", color="Impact Score"),
                                color_continuous_scale=[[0, 'red'], [0.5, 'white'], [1, 'green']],
                                range_color=[-3, 3], aspect="auto", height=200)
            st.plotly_chart(fig_macro, use_container_width=True)

            # --- COMPARISON WITH BOTTOM-UP VIEW ---
            with st.expander("⚖️ Compare Macro vs. Bottom-Up Dashboard View"):
                st.markdown("Comparison between **PDF-based Macro Views** and your **Company-based Bottom-Up Averages**.")
                # Get Dashboard Sector Averages from existing sector_summary
                bottom_up_avg = sector_summary[["UKR_AVG_SCORE", "IRAN_AVG_SCORE"]].rename(columns={"UKR_AVG_SCORE": "Bottom-Up 🇺🇦", "IRAN_AVG_SCORE": "Bottom-Up 🇮🇷"})
                
                # Merge with Macro heatmap
                comparison_df = macro_heatmap_df.copy().rename(columns={"🇺🇦 Ukraine 2022": "Macro 🇺🇦", "🇮🇷 Iran 2026": "Macro 🇮🇷"})
                final_comp = comparison_df.join(bottom_up_avg, how='inner')
                
                st.dataframe(final_comp.style.applymap(color_scores_simple), use_container_width=True)
                st.info("💡 **Insights:** If Macro > Bottom-Up, the market reports are more optimistic than the specific companies in your list. If Macro < Bottom-Up, the specific companies are showing more resilience than the sector theory suggests.")
            
            # --- MACRO CHARTS SECTION ---
            st.divider()
            st.subheader("📈 Macro Asset Performance Around Shocks")
            asset_dir = "." # Images must be copied to the local folder for deployment
            
            # Line 1: 3 images
            row1_c1, row1_c2, row1_c3 = st.columns(3)
            with row1_c1: st.image(os.path.join(asset_dir, "WTI_price_around_geopolitical_shocks.png"), caption="WTI Crude Oil")
            with row1_c2: st.image(os.path.join(asset_dir, "SPY_price_around_geopolitical shocks.png"), caption="S&P 500 (SPY)")
            with row1_c3: st.image(os.path.join(asset_dir, "SX5E_around_geopolitical_shocks.png"), caption="Euro Stoxx 50 (SX5E)")
            
            # Line 2: Remaining 2 images
            row2_c1, row2_c2, row2_c3 = st.columns(3)
            with row2_c1: st.image(os.path.join(asset_dir, "Gold_price_around_geopolitcal_shocks.png"), caption="Gold Price")
            with row2_c2: st.image(os.path.join(asset_dir, "DollarIndex_around_geopolitical_shocks.png"), caption="US Dollar Index (DXY)")

            st.divider()
            st.subheader("🔍 Single Stock Mentions")
            mentions_raw = geopol_data['stock_mentions']
            mentions_df = pd.DataFrame(mentions_raw)
            
            # --- INTERACTIVE FILTER FOR FACTORS ---
            st.markdown("🎯 **Keyword Filter (e.g. 'oil', 'semis', 'supply')**")
            keyword_query = st.text_input("Search through factors and reasons:", placeholder="Type a keyword to filter the stock mentions below...", key="mention_keyword")
            
            if keyword_query:
                # Fuzzy/Partial Match Logic
                q = keyword_query.lower()
                def match_mention(row):
                    factors = " ".join(row['factors_exposed_to']).lower()
                    reason = row['reason_mentioned'].lower()
                    return q in factors or q in reason
                
                mentions_df = mentions_df[mentions_df.apply(match_mention, axis=1)]
                st.info(f"Showing {len(mentions_df)} mentions matching '{keyword_query}'")

            # --- SORTING FOR MENTIONS ---
            ms_c1, ms_c2 = st.columns([2, 1])
            with ms_c1:
                mention_sort_col = st.selectbox("Sort Table By:", ["company_name", "ticker", "mentioned_date"], index=2, key="ms_sort")
            with ms_c2:
                mention_sort_order = st.radio("Sort Order:", ["Desc", "Asc"], horizontal=True, key="ms_order")
            
            mentions_df = mentions_df.sort_values(by=mention_sort_col, ascending=(mention_sort_order == "Asc"))

            # 2. Refined Column Display
            cols_to_show = ["company_name", "ticker", "mentioned_date", "reason_mentioned", "factors_exposed_to", "original_sentences", "source_file"]
            display_df = mentions_df[cols_to_show].copy()
            display_df['factors_exposed_to'] = display_df['factors_exposed_to'].apply(lambda x: ", ".join(x))
            
            # Use Pandas Styler to force widths and wrapping
            st.markdown("""
                <style>
                .styled-table td {
                    white-space: normal !important;
                    word-wrap: break-word !important;
                    min-width: 100px;
                }
                </style>
            """, unsafe_allow_html=True)

            styled_table = display_df.style.set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#f0f2f6'), ('text-align', 'left'), ('font-weight', 'bold')]},
                {'selector': 'td', 'props': [('vertical-align', 'top'), ('padding', '10px'), ('border', '1px solid #eee')]},
                # Set specific min-widths for long text columns
                {'selector': '.col3', 'props': [('min-width', '250px')]}, # Reason
                {'selector': '.col5', 'props': [('min-width', '450px')]}, # Original Sentences
                {'selector': '.col0', 'props': [('min-width', '150px')]}, # Company
            ])

            st.write(styled_table.to_html(classes='styled-table'), unsafe_allow_html=True)
        else:
            st.warning("Geopolitical analysis file not found.")

    with tab_search:
        st.subheader("🧠 AI Semantic Intelligence Search")
        st.markdown("Type a query below to search for companies based on specific risks or themes.")
        search_prompt = st.text_input("Semantic Query", placeholder="e.g., 'companies exposed to supply chain risks in Iran conflict'")
        if search_prompt:
            results = perform_semantic_search(search_prompt)
            if not results.empty:
                for _, res in results.iterrows():
                    with st.container(border=True):
                        st.markdown(f"**{res['company']}** ({res['ticker']}) | Relevance: {1-res.get('_distance', 0):.2f}")
                        st.markdown(f"**Topic:** {res['topic']}")
                        st.write(f"**Details:** {res['details']}")
                        st.info(f"**Context:** {res['context']}")
            else: st.write("No matches found.")

# --- AI RESEARCH ASSISTANT (RIGHT SIDEBAR) ---
with chat_col:
    if os.path.exists("Iran_US_impact.png"):
        st.image("Iran_US_impact.png", caption="Potential US / Iran Impact Model", use_container_width=True)
    
    if os.path.exists("Iran_Ukraine_diff.png"):
        st.image("Iran_Ukraine_diff.png", caption="Iran vs. Ukraine: Key Differences", use_container_width=True)
    
    st.subheader("💬 AI Analyst")
    if "messages" not in st.session_state: st.session_state.messages = []
    chat_container = st.container(height=600)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"): st.markdown(prompt)
        try:
            sector_summary_str = sector_summary.to_csv()
            context_str = f"Sector Averages:\n{sector_summary_str}\n\n"
            # Note: selected_company only defined in tab_dashboard scope, using try/except
            try:
                if selected_company:
                    d = get_company_details(selected_company)
                    if d: context_str += f"Active Company ({selected_company}):\n{json.dumps(d, indent=2)}\n\n"
            except: pass
            response = client.models.generate_content(model=MODEL_ID, contents=f"You are a financial analyst. Context:\n{context_str}\n\nUser Question: {prompt}\n\nExplain rationale and ask for feedback.")
            res_text = response.text
        except Exception as e: res_text = f"Error: {e}"
        with chat_container:
            with st.chat_message("assistant"): st.markdown(res_text)
        st.session_state.messages.append({"role": "assistant", "content": res_text})
