import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="מודל כלונס מלא – עומס מצטבר ולחץ בקצה", layout="wide")
st.title("🏗️ מודל גיאומכני רב-שכבתי – עומס מצטבר ולחץ בקצה הכלונס (V6)")
st.markdown("---")

# פונקציית חישוב מאמץ הגזירה המקומי לפי התזוזה (אלסטי -> ריכוך -> משתייר)
def calc_shear_stress(w, t_peak, t_res, w_p, w_r):
    if w <= 0 or t_peak <= 0:
        return 0.0, "ללא תזוזה"
    elif w <= w_p:
        return (t_peak / w_p) * w, "אלסטי"
    elif w <= w_r:
        slope = (t_peak - t_res) / max((w_r - w_p), 1e-6)
        return t_peak - slope * (w - w_p), "ריכוך (ירידה)"
    else:
        return t_res, "משתייר (Residual)"

# פונקציית הרצה בודדת של המודל עבור שקיעת בסיס נתונה
def run_pile_model(layers_df, E, D, B, w_base):
    A = np.pi * (D**2) / 4.0
    U = np.pi * D
    num_layers = len(layers_df)
    depths = [0.0] + layers_df["עומק תחתית שכבה (מ')"].tolist()
    
    w_node = np.zeros(num_layers + 1)
    P_node = np.zeros(num_layers + 1)
    t_peak_seg = np.zeros(num_layers)
    t_res_seg = np.zeros(num_layers)
    t_mob_seg = np.zeros(num_layers)
    F_fric_seg = np.zeros(num_layers)
    state_seg = [""] * num_layers

    w_node[num_layers] = w_base
    P_node[num_layers] = B * np.sqrt(w_base) if w_base > 0 else 0.0

    for i in range(num_layers - 1, -1, -1):
        dz = max(depths[i + 1] - depths[i], 0.01)
        sig_h = layers_df.loc[i, "לחץ פרסיומטר σ'h (kPa)"]
        d_peak = layers_df.loc[i, "זווית חיכוך שיא δ_peak (°)"]
        w_p = layers_df.loc[i, "תזוזת שיא w_peak (מ\"מ)"] / 1000.0
        d_res = layers_df.loc[i, "זווית משתיירת δ_res (°)"]
        w_r = layers_df.loc[i, "תזוזת התייצבות w_res (מ\"מ)"] / 1000.0

        t_p = sig_h * np.tan(np.radians(d_peak))
        t_r = sig_h * np.tan(np.radians(d_res))

        w_top_est = w_node[i + 1] + (P_node[i + 1] * dz) / (E * A)
        w_mid = 0.5 * (w_node[i + 1] + w_top_est)

        t_mob, state = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)
        F_fric = U * dz * t_mob

        P_node[i] = P_node[i + 1] + F_fric
        P_mean = 0.5 * (P_node[i] + P_node[i + 1])
        w_node[i] = w_node[i + 1] + (P_mean * dz) / (E * A)

        t_peak_seg[i] = t_p
        t_res_seg[i] = t_r
        t_mob_seg[i] = t_mob
        F_fric_seg[i] = F_fric
        state_seg[i] = state

    # חישוב עומס חיכוך מצטבר מלמעלה למטה ולחץ בקצה
    cum_fric_seg = np.cumsum(F_fric_seg)
    q_base = P_node[-1] / A

    return depths, w_node, P_node, t_peak_seg, t_res_seg, t_mob_seg, F_fric_seg, cum_fric_seg, q_base, state_seg

col_inputs, col_results = st.columns([1.25, 1.75])

with col_inputs:
    st.header("⚙️ 1. פרמטרים מבניים ותנאי קצה")
    c1, c2 = st.columns(2)
    with c1:
        E = st.number_input("מודול אלסטיות חומר E (kPa)", value=30000000, step=1000000)
        D = st.number_input("קוטר הכלונס D (מטר)", min_value=0.2, max_value=3.0, value=0.6, step=0.05)
    with c2:
        B = st.number_input("מקדם בסיס פרבולי B (kN/m^0.5)", value=150.0, step=10.0)
        w_base_mm = st.slider("נקודת הרצה נוכחית – שקיעת בסיס (מ\"מ)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)

    st.markdown("---")
    st.header("📂 2. נתוני השכבות (טעינת קובץ או עריכה)")
    
    default_layers = pd.DataFrame({
        "עומק תחתית שכבה (מ')": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "לחץ פרסיומטר σ'h (kPa)": [90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 175.0, 190.0],
        "זווית חיכוך שיא δ_peak (°)": [32.0, 32.0, 32.0, 30.0, 30.0, 30.0, 33.0, 33.0, 34.0, 34.0],
        "תזוזת שיא w_peak (מ\"מ)": [2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0],
        "זווית משתיירת δ_res (°)": [26.0, 26.0, 26.0, 24.0, 24.0, 24.0, 27.0, 27.0, 28.0, 28.0],
        "תזוזת התייצבות w_res (מ\"מ)": [8.0, 8.0, 10.0, 10.0, 12.0, 12.0, 9.0, 9.0, 9.0, 9.0]
    })

    csv_template = default_layers.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 הורד תבנית CSV ריקה להזנת נתוני קידוח",
        data=csv_template,
        file_name="pile_layers_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("או העלה קובץ נתונים מוכן (CSV / XLSX):", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file)
            st.success("הקובץ נטען בהצלחה! ניתן לערוך ערכים נוספים בטבלה למטה.")
        except Exception as e:
            st.error(f"שגיאה בקריאת הקובץ: {e}")
            input_df = default_layers
    else:
        input_df = default_layers

    edited_df = st.data_editor(input_df, num_rows="dynamic", use_container_width=True, hide_index=True)

layers_df = edited_df.dropna().sort_values(by="עומק תחתית שכבה (מ')").reset_index(drop=True)

if len(layers_df) > 0:
    depths, w_node, P_node, t_peak_seg, t_res_seg, t_mob_seg, F_fric_seg, cum_fric_seg, q_base, state_seg = run_pile_model(
        layers_df, E, D, B, w_base_mm / 1000.0
    )

    w_base_sweep = np.linspace(0.0, 0.05, 60)
    P0_curve, w0_curve, Pbase_curve, qbase_curve, cum_fric_curve = [], [], [], [], []
    for wb in w_base_sweep:
        _, wn, Pn, _, _, _, _, cf_seg, qb, _ = run_pile_model(layers_df, E, D, B, wb)
        w0_curve.append(wn[0] * 1000.0)
        P0_curve.append(Pn[0])
        Pbase_curve.append(Pn[-1])
        qbase_curve.append(qb)
        cum_fric_curve.append(cf_seg[-1])

    with col_results:
        st.header("📊 תוצאות המודל: עומס מצטבר ולחץ בקצה")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("עומס בראש P₀", f"{P_node[0]:,.1f} kN")
        k2.metric("תזוזה בראש w₀", f"{w_node[0]*1000:.2f} מ\"מ")
        k3.metric("עומס בקצה P_base", f"{P_node[-1]:,.1f} kN")
        k4.metric("לחץ בקצה q_base", f"{q_base:,.1f} kPa")

        results_df = pd.DataFrame({
            "שכבה": range(1, len(layers_df) + 1),
            "עומק (מ')": [f"{depths[i]:.1f} - {depths[i+1]:.1f}" for i in range(len(layers_df))],
            "תזוזה w (מ\"מ)": w_node[:-1] * 1000.0,
            "מאמץ בפועל t_mob (kPa)": t_mob_seg,
            "כוח שכבה (kN)": F_fric_seg,
            "חיכוך מצטבר ΣF (kN)": cum_fric_seg,
            "כוח צירי נותר P (kN)": P_node[:-1],
            "מצב השכבה": state_seg
        })

        report_csv = results_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📄 הורד דו\"ח תוצאות מלא לאקסל (CSV)",
            data=report_csv,
            file_name=f"pile_analysis_report_D{D}_w{w_base_mm}mm.csv",
            mime="text/csv"
        )

        tab1, tab2 = st.tabs(["📈 עומס מצטבר ולחץ בקצה", "📋 טבלת שכבות ופרופיל עומק"])

        with tab1:
            fig_ls, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(10, 4.2))
            
            # גרף שמאלי: חלוקת העומס הכולל בין חיכוך מצטבר לעומס בקצה
            ax_top.plot(w0_curve, P0_curve, label='עומס כולל בראש (P₀)', color='#1f497d', linewidth=2.5)
            ax_top.plot(w0_curve, cum_fric_curve, '-.', label='חיכוך מעטפת מצטבר (ΣF)', color='green', linewidth=2)
            ax_top.plot(w0_curve, Pbase_curve, '--', label='עומס שמגיע לקצה (P_base)', color='#d9534f', linewidth=2)
            ax_top.scatter([w_node[0] * 1000.0], [P_node[0]], color='black', s=70, zorder=5)
            ax_top.set_xlabel("שקיעה בראש הכלונס w₀ (מ\"מ)")
            ax_top.set_ylabel("כוח (kN)")
            ax_top.set_title("מאזן כוחות: חיכוך מצטבר מול קצה")
            ax_top.grid(True, linestyle='--', alpha=0.5)
            ax_top.legend(fontsize=8)

            # גרף ימני: התפתחות הלחץ בקצה הכלונס (kPa) כפונקציה של העומס בראש
            ax_bot.plot(P0_curve, qbase_curve, color='#b83227', linewidth=2.5)
            ax_bot.scatter([P_node[0]], [q_base], color='green', s=80, zorder=5, label='נקודת הרצה נוכחית')
            ax_bot.set_xlabel("עומס חיצוני בראש P₀ (kN)")
            ax_bot.set_ylabel("לחץ בקצה הכלונס q_base (kPa)")
            ax_bot.set_title("התפתחות הלחץ בקצה עם עליית העומס")
            ax_bot.grid(True, linestyle='--', alpha=0.5)
            ax_bot.legend(fontsize=8)

            plt.tight_layout()
            st.pyplot(fig_ls)

        with tab2:
            st.dataframe(results_df.style.format({
                "תזוזה w (מ\"מ)": "{:.2f}",
                "מאמץ בפועל t_mob (kPa)": "{:.1f}",
                "כוח שכבה (kN)": "{:.1f}",
                "חיכוך מצטבר ΣF (kN)": "{:,.1f}",
                "כוח צירי נותר P (kN)": "{:,.1f}"
            }), use_container_width=True, hide_index=True)

            fig_prof, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))
            mid_depths = [0.5 * (depths[i] + depths[i+1]) for i in range(len(layers_df))]

            ax1.plot(w_node * 1000.0, depths, marker='o', color='#2e75b6')
            ax1.set_title("תזוזות (מ\"מ)")
            ax1.set_xlabel("w (מ\"מ)")
            ax1.set_ylabel("עומק (מטר)")
            ax1.invert_yaxis()
            ax1.grid(True, linestyle='--', alpha=0.5)

            ax2.plot(cum_fric_seg, layers_df["עומק תחתית שכבה (מ')"], marker='d', color='green')
            ax2.set_title("עומס חיכוך מצטבר (kN)")
            ax2.set_xlabel("ΣF (kN)")
            ax2.invert_yaxis()
            ax2.grid(True, linestyle='--', alpha=0.5)

            ax3.plot(P_node, depths, marker='^', color='#1f497d')
            ax3.set_title("כוח צירי נותר (kN)")
            ax3.set_xlabel("P (kN)")
            ax3.invert_yaxis()
            ax3.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig_prof)