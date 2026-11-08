import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration / הגדרות עמוד
st.set_page_config(page_title="Geotechnical Pile Model / מודל גיאומכני לכלונסאות", layout="wide")

st.title("📊 Geotechnical Finite Difference Model - Multi-Layer Soil & Spring Stiffness")
st.subheader("מודל גיאומכני להפרשים סופיים - כלונס בשכבות קרקע עם קשיחות קפיצים")
st.markdown("---")

# Layout: Inputs on the right (Hebrew friendly), Results on the left
col_results, col_inputs = st.columns([2, 1])

with col_inputs:
    st.header("⚙️ Input Parameters / פרמטרים וקלטים")
    
    # Structural parameters / פרמטרים מבניים
    st.subheader("🏗️ Pile Properties / מאפייני הכלונס")
    E = st.number_input("Pile Elastic Modulus / מודול אלסטיות E (kPa)", value=30000000, step=1000000)
    D = st.slider("Pile Diameter / קוטר הכלונס D (m)", min_value=0.3, max_value=2.0, value=0.6, step=0.05)
    B = st.number_input("Base Coefficient / מקדם קשיחות בסיס b (kN/m^0.5)", value=150.0, step=10.0)
    
    A = np.pi * (D**2) / 4
    U = np.pi * D
    dz = 1.0  # constant segment length / אורך מקטע קבוע
    
    st.markdown("---")
    # Base settlement slider / שקיעת הבסיס
    w_base_input = st.slider("Base Settlement / שקיעת הבסיס w_base (m)", min_value=0.0, max_value=0.03, value=0.005, step=0.0005, format="%.4f")
    
    st.markdown("---")
    st.subheader("🌍 Soil Layers Boundaries / הגדרת עומק השכבות")
    
    # Manual depth input for layers / הזנת עומקים ידנית
    layer1_end = st.number_input("Layer 1 Bottom Depth / עומק תחתית שכבה 1 (m)", min_value=1.0, max_value=9.0, value=3.0, step=1.0)
    layer2_end = st.number_input("Layer 2 Bottom Depth / עומק תחתית שכבה 2 (m)", min_value=layer1_end+1.0, max_value=9.5, value=7.0, step=1.0)
    
    st.markdown("**Select Soil Type per Layer / בחר סוג קרקע לכל שכבה:**")
    layer_top = st.selectbox("Layer 1 Soil / סוג קרקע שכבה 1", ["Loose Sand / חול רך", "Dense Sand / חול צפוף", "Clay / חרסית"], index=0)
    layer_mid = st.selectbox("Layer 2 Soil / סוג קרקע שכבה 2", ["Loose Sand / חול רך", "Dense Sand / חול צפוף", "Clay / חרסית"], index=1)
    layer_bot = st.selectbox("Layer 3 Soil / סוג קרקע שכבה 3", ["Loose Sand / חול רך", "Dense Sand / חול צפוף", "Clay / חרסית"], index=1)
    
    # Dynamic assignment of t_max and spring reference stiffness (k_spring in kN/m3) based on selection
    def get_soil_properties(soil_type, depth):
        if depth == 0:
            return 0, 0
        if "Dense Sand" in soil_type:
            return 100 + (10 - depth) * 2, 40000  # High t_max, high subgrade modulus
        elif "Loose Sand" in soil_type:
            return 65 + depth * 2, 15000       # Lower t_max, lower subgrade modulus
        else: # Clay / חרסית
            return 80, 25000                   # Constant cohesion behavior
            
    t_max = {}
    k_soil = {} # subgrade modulus (kN/m3)
    
    for depth in range(11):
        if depth == 0:
            t_max[depth] = 0
            k_soil[depth] = 0
        elif depth <= layer1_end:
            t_max[depth], k_soil[depth] = get_soil_properties(layer_top, depth)
        elif depth <= layer2_end:
            t_max[depth], k_soil[depth] = get_soil_properties(layer_mid, depth)
        else:
            t_max[depth], k_soil[depth] = get_soil_properties(layer_bot, depth)

# Model computation core / פונקציית ליבה של החישוב
def run_pile_model(w_b):
    w_v = {}
    F_f = {}
    P_v = {}
    k_spring_eff = {} # Calculated local spring stiffness (kN/m)
    
    for node in range(10, -1, -1):
        if node == 10:
            w_v[node] = w_b
            F_f[node] = U * dz * t_max[node]
            P_v[node] = B * np.sqrt(w_b) if w_b > 0 else 0.0
            k_spring_eff[node] = (k_soil[node] * U * dz) if w_b == 0 else (F_f[node] / w_b if w_b > 0 else 0)
        else:
            prev = node + 1
            w_v[node] = w_v[prev] + (P_v[prev] * dz) / (E * A)
            F_f[node] = U * dz * t_max[node] if node > 0 else 0
            P_v[node] = P_v[prev] + F_f[prev]
            k_spring_eff[node] = (k_soil[node] * U * dz) if w_v[node] == 0 else (F_f[node] / w_v[node] if w_v[node] > 0 else 0)
            
    return w_v, F_f, P_v, k_spring_eff

# 1. Compute for the current selected point / חישוב הנקודה הנבחרת
w_actual, F_actual, P_actual, k_spring_actual = run_pile_model(w_base_input)

# 2. Build Load-Settlement Curve / בניית עקומת עומס-שקיעה (שורות 73-74 המתוקנות)
w_base_range = np.linspace(0, 0.03, 100)
top_loads = []
top_settlements = []

for wb in w_base_range:
    w_res, _, P_res, _ = run_pile_model(wb)
    top_loads.append(P_res[0])
    top_settlements.append(w_res[0] * 1000)

# Build results dataframe / טבלת נתונים
df_results = pd.DataFrame({
    "Node / צומת (i)": list(range(11)),
    "Depth / עומק z (m)": [i * 1.0 for i in range(11)],
    "Settlement / שקיעה w_i (m)": [w_actual[i] for i in range(11)],
    "t_max (kPa)": [t_max[i] for i in range(11)],
    "Spring Stiffness / קשיחות קפיץ K_sp (kN/m)": [k_spring_actual[i] for i in range(11)],
    "Friction Force / כוח חיכוך F_fric (kN)": [F_actual[i] for i in range(11)],
    "Internal Force / כוח פנימי P_i (kN)": [P_actual[i] for i in range(11)]
})

with col_results:
    st.header("📊 Model Analysis Results / תוצאות המודל")
    
    # Metrics / מדדים מהירים
    m1, m2, m3 = st.columns(3)
    m1.metric("Top Load / עומס בראש P₀", f"{P_actual[0]:,.1f} kN")
    m2.metric("Top Settlement / שקיעה בראש w₀", f"{w_actual[0]*1000:.2f} mm")
    m3.metric("Base Force / כוח בבסיס P_base", f"{P_actual[10]:,.1f} kN")
    
    # Graphs - Three columns / שלושה טורי גרפים
    g_col1, g_col2, g_col3 = st.columns(3)
    
    with g_col1:
        st.markdown("### 📈 Load-Settlement / עומס-שקיעה")
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.plot(top_loads, top_settlements, color='#C00000', linewidth=2.5)
        ax1.plot(P_actual[0], w_actual[0]*1000, marker='X', color='black', markersize=10)
        ax1.set_xlabel("Top Load P₀ (kN)")
        ax1.set_ylabel("Top Settlement w₀ (mm)")
        ax1.invert_yaxis()
        ax1.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig1)

    with g_col2:
        st.markdown("### 📉 Force Distribution / פיזור מאמצים")
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.plot(df_results["Internal Force / כוח פנימי P_i (kN)"], df_results["Depth / עומק z (m)"], marker='o', color='#1F497D', linewidth=2)
        ax2.set_xlabel("Internal Force P_i (kN)")
        ax2.set_ylabel("Depth z (m)")
        ax2.invert_yaxis()
        ax2.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig2)

    with g_col3:
        st.markdown("### 📊 Spring Stiffness / קשיחות קפיץ")
        fig3, ax3 = plt.subplots(figsize=(4, 4))
        ax3.plot(df_results["Spring Stiffness / קשיחות קפיץ K_sp (kN/m)"], df_results["Depth / עומק z (m)"], marker='s', color='#2E7D32', linewidth=2)
        ax3.set_xlabel("Spring Stiffness K_sp (kN/m)")
        ax3.set_ylabel("Depth z (m)")
        ax3.invert_yaxis()
        ax3.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig3)
        
    st.markdown("### Detailed Element Table / טבלת אלמנטים מפורטת")
    st.dataframe(df_results.style.format({
        "Settlement / שקיעה w_i (m)": "{:.4f}",
        "Spring Stiffness / קשיחות קפיץ K_sp (kN/m)": "{:,.1f}",
        "Friction Force / כוח חיכוך F_fric (kN)": "{:,.1f}",
        "Internal Force / כוח פנימי P_i (kN)": "{:,.1f}"
    }), use_container_width=True)

    # 💾 Export to CSV Button / כפתור ייצוא ל-CSV
    st.markdown("---")
    st.subheader("💾 Export Data / ייצוא נתונים")
    
    csv_data = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results as CSV / הורד תוצאות כקובץ CSV",
        data=csv_data,
        file_name="Pile_Model_Advanced.csv",
        mime="text/csv"
    )
report_csv = results_df.to_csv(index=False).encode('utf-8-sig')
st.download_button(
label="📄 הורד דו"ח תוצאות מלא לאקסל (CSV)",
data=report_csv,
file_name=f"pile_topdown_report_D{D}_w{w0_mm}mm.csv",
mime="text/csv"
)
tab1, tab2 = st.tabs(["📈 עקומת עומס-שקיעה מלאה (מ-0 עד כשל)", "📋 התקדמות החזית לעומק"])
with tab1:
fig_ls, ax_ls = plt.subplots(figsize=(8.5, 4.2))
ax_ls.plot(w0_curve, P0_curve, label='עומס כולל בראש הכלונס (P₀)', color='#1f497d', linewidth=2.5)
ax_ls.plot(w0_curve, cum_fric_curve, '-.', label='חיכוך מעטפת מצטבר (ΣF)', color='green', linewidth=2)
ax_ls.plot(w0_curve, Pbase_curve, '--', label='עומס שמגיע לקצה (P_base)', color='#d9534f', linewidth=2)
ax_ls.scatter([w0_mm], [float(P_node.item(0))], color='black', s=85, zorder=5, label='נקודת הרצה נוכחית')
ax_ls.set_xlabel("תזוזה בראש הכלונס w₀ (מ"מ)")
ax_ls.set_ylabel("כוח (kN)")
ax_ls.set_title("עקומת עומס-שקיעה מלאה – משלב א' (גיוס מעטפת) לשלב ב' (גיוס קצה)")
ax_ls.grid(True, linestyle='--', alpha=0.5)
ax_ls.legend()
st.pyplot(fig_ls)
with tab2:
st.dataframe(results_df.style.format({
"תזוזה w (מ"מ)": "{:.2f}",
"מאמץ בפועל t_mob (kPa)": "{:.1f}",
"כוח שכבה (kN)": "{:.1f}",
"חיכוך מצטבר ΣF (kN)": "{:,.1f}",
"כוח צירי נותר P (kN)": "{:,.1f}"
}), use_container_width=True, hide_index=True)

END OF FILE
