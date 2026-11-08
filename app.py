import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

st.set_page_config(page_title="מודל כלונס מלמעלה למטה (Top-Down)", layout="wide")
st.title("🏗️ מודל גיאומכני מלמעלה למטה (Top-Down) – גיוס הדרגתי מהראש לקצה (V7)")
st.markdown("---")

# פונקציית מאמץ הגזירה המקומי לפי התזוזה (אלסטי -> ריכוך -> משתייר)
def calc_shear_stress(w, t_peak, t_res, w_p, w_r):
    if w <= 1e-9 or t_peak <= 0:
        return 0.0, "נייח (עומס טרם הגיע)"
    elif w <= w_p:
        return (t_peak / w_p) * w, "אלסטי (מגייס חיכוך)"
    elif w <= w_r:
        slope = (t_peak - t_res) / max((w_r - w_p), 1e-6)
        return t_peak - slope * (w - w_p), "ריכוך (ירידה למשתייר)"
    else:
        return t_res, "משתייר (Residual)"

# סימולציה שיורדת מלמעלה למטה (מראש הכלונס לקצה) עבור שקיעת ראש נתונה w0
def shoot_top_down(w0, layers_df, E, D, B):
    A = np.pi * (D**2) / 4.0
    U = np.pi * D
    num_layers = len(layers_df)
    depths = [0.0] + layers_df["עומק תחתית שכבה (מ')"].tolist()

    # שלב א': בדיקת שיווי משקל כאשר הקצה נייח (w_base = 0)
    # מתחילים מלמטה עם w=0 ועולים, או בודקים אם w0 מספיק להגיע לקצה
    w_node = np.zeros(num_layers + 1)
    P_node = np.zeros(num_layers + 1)
    t_mob_seg = np.zeros(num_layers)
    F_fric_seg = np.zeros(num_layers)
    state_seg = [""] * num_layers

    # חישוב שקיעת הראש המינימלית שבה החזית מגיעה בדיוק לקצה (w_base = 0)
    w_crit = np.zeros(num_layers + 1)
    P_crit = np.zeros(num_layers + 1)
    for i in range(num_layers - 1, -1, -1):
        dz = max(depths[i + 1] - depths[i], 0.01)
        sig_h = layers_df.loc[i, "לחץ פרסיומטר σ'h (kPa)"]
        t_p = sig_h * np.tan(np.radians(layers_df.loc[i, "זווית חיכוך שיא δ_peak (°)"]))
        t_r = sig_h * np.tan(np.radians(layers_df.loc[i, "זווית משתיירת δ_res (°)"]))
        w_p = layers_df.loc[i, "תזוזת שיא w_peak (מ\"מ)"] / 1000.0
        w_r = layers_df.loc[i, "תזוזת התייצבות w_res (מ\"מ)"] / 1000.0

        # הערכת תזוזה באמצע המקטע
        w_est = w_crit[i + 1] + (P_crit[i + 1] * dz) / (E * A) + 0.5 * w_p
        t_m, _ = calc_shear_stress(0.5 * (w_crit[i + 1] + w_est), t_p, t_r, w_p, w_r)
        F_f = U * dz * t_m
        P_crit[i] = P_crit[i + 1] + F_f
        w_crit[i] = w_crit[i + 1] + (0.5 * (P_crit[i] + P_crit[i + 1]) * dz) / (E * A)

    w0_limit = w_crit

    # בדיקה: האם אנחנו בשלב א' (החזית טרם הגיעה לקצה) או בשלב ב' (הקצה מגויס)?
    if w0 <= w0_limit:
        # שלב א': ירידה מלמעלה למטה – מוצאים באיזה עומק התזוזה והכוח מתאפסים
        w_node = w0
        # פתרון איטרטיבי למציאת P0 שמתאים ל-w0 כאשר התחתית חופשית (P_base = 0)
        def residual_stage_a(P0_guess):
            w_tmp = np.zeros(num_layers + 1)
            P_tmp = np.zeros(num_layers + 1)
            w_tmp = w0
            P_tmp = P0_guess
            for i in range(num_layers):
                if P_tmp[i] <= 1e-6 or w_tmp[i] <= 1e-9:
                    w_tmp[i + 1] = 0.0
                    P_tmp[i + 1] = 0.0
                    continue
                dz = max(depths[i + 1] - depths[i], 0.01)
                sig_h = layers_df.loc[i, "לחץ פרסיומטר σ'h (kPa)"]
                t_p = sig_h * np.tan(np.radians(layers_df.loc[i, "זווית חיכוך שיא δ_peak (°)"]))
                t_r = sig_h * np.tan(np.radians(layers_df.loc[i, "זווית משתיירת δ_res (°)"]))
                w_p = layers_df.loc[i, "תזוזת שיא w_peak (מ\"מ)"] / 1000.0
                w_r = layers_df.loc[i, "תזוזת התייצבות w_res (מ\"מ)"] / 1000.0

                w_bot_est = max(w_tmp[i] - (P_tmp[i] * dz) / (E * A), 0.0)
                w_mid = 0.5 * (w_tmp[i] + w_bot_est)
                t_m, _ = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)
                F_f = U * dz * t_m

                if F_f >= P_tmp[i]:
                    # הכוח נספג לחלוטין בתוך שכבה זו!
                    P_tmp[i + 1] = 0.0
                    w_tmp[i + 1] = 0.0
                else:
                    P_tmp[i + 1] = P_tmp[i] - F_f
                    P_mean = 0.5 * (P_tmp[i] + P_tmp[i + 1])
                    w_tmp[i + 1] = max(w_tmp[i] - (P_mean * dz) / (E * A), 0.0)
            return P_tmp[-1]

        try:
            P0_solved = brentq(residual_stage_a, 0.0, P_crit * 1.5)
        except:
            P0_solved = P_crit * (w0 / max(w0_limit, 1e-6))

        # הרצה סופית מלמעלה למטה עם P0_solved
        w_node = w0
        P_node = P0_solved
        for i in range(num_layers):
            if P_node[i] <= 1e-6 or w_node[i] <= 1e-9:
                w_node[i + 1] = 0.0
                P_node[i + 1] = 0.0
                t_mob_seg[i] = 0.0
                F_fric_seg[i] = 0.0
                state_seg[i] = "נייח (עומס טרם הגיע)"
                continue
            dz = max(depths[i + 1] - depths[i], 0.01)
            sig_h = layers_df.loc[i, "לחץ פרסיומטר σ'h (kPa)"]
            t_p = sig_h * np.tan(np.radians(layers_df.loc[i, "זווית חיכוך שיא δ_peak (°)"]))
            t_r = sig_h * np.tan(np.radians(layers_df.loc[i, "זווית משתיירת δ_res (°)"]))
            w_p = layers_df.loc[i, "תזוזת שיא w_peak (מ\"מ)"] / 1000.0
            w_r = layers_df.loc[i, "תזוזת התייצבות w_res (מ\"מ)"] / 1000.0

            w_bot_est = max(w_node[i] - (P_node[i] * dz) / (E * A), 0.0)
            w_mid = 0.5 * (w_node[i] + w_bot_est)
            t_m, st_label = calc_shear_stress(w_mid, t_p, t_r, w_p, w_r)
            F_f = U * dz * t_m

            if F_f >= P_node[i]:
                F_fric_seg[i] = P_node[i]
                t_mob_seg[i] = P_node[i] / (U * dz)
                P_node[i + 1] = 0.0
                w_node[i + 1] = 0.0
                state_seg[i] = "בלימת החזית (כאן הכוח מתאפס)"
            else:
                F_fric_seg[i] = F_f
                t_mob_seg[i] = t_m
                P_node[i + 1] = P_node[i] - F_f
                P_mean = 0.5 * (P_node[i] + P_node[i + 1])
                w_node[i + 1] = max(w_node[i] - (P_mean * dz) / (E * A), 0.0)
                state_seg[i] = st_label

        stage_name = "שלב א': גיוס הדרגתי מלמעלה (הקצה עדיין נייח, P_base = 0)"
    else:
        # שלב ב': החזית עברה את כל המעטפת והקצה התחתון מגויס (w_base > 0)
        def eval_from_wbase(wb):
            wn = np.zeros(num_layers + 1)
            Pn = np.zeros(num_layers + 1)
            tm = np.zeros(num_layers)
            Ff = np.zeros(num_layers)
            ss = [""] * num_layers
            wn[num_layers] = wb
            Pn[num_layers] = B * np.sqrt(wb) if wb > 0 else 0.0
            for i in range(num_layers - 1, -1, -1):
                dz = max(depths[i + 1] - depths[i], 0.01)
                sig_h = layers_df.loc[i, "לחץ פרסיומטר σ'h (kPa)"]
                t_p = sig_h * np.tan(np.radians(layers_df.loc[i, "זווית חיכוך שיא δ_peak (°)"]))
                t_r = sig_h * np.tan(np.radians(layers_df.loc[i, "זווית משתיירת δ_res (°)"]))
                w_p = layers_df.loc[i, "תזוזת שיא w_peak (מ\"מ)"] / 1000.0
                w_r = layers_df.loc[i, "תזוזת התייצבות w_res (מ\"מ)"] / 1000.0

                w_top_est = wn[i + 1] + (Pn[i + 1] * dz) / (E * A)
                t_mob, st_l = calc_shear_stress(0.5 * (wn[i + 1] + w_top_est), t_p, t_r, w_p, w_r)
                F_f = U * dz * t_mob
                Pn[i] = Pn[i + 1] + F_f
                wn[i] = wn[i + 1] + (0.5 * (Pn[i] + Pn[i + 1]) * dz) / (E * A)
                tm[i], Ff[i], ss[i] = t_mob, F_f, st_l
            return wn, Pn, tm, Ff, ss

        wb_solved = brentq(lambda wb: eval_from_wbase(wb) - w0, 0.0, 0.2)
        w_node, P_node, t_mob_seg, F_fric_seg, state_seg = eval_from_wbase(wb_solved)
        stage_name = "שלב ב': הקצה מגויס (תגובת בסיס פרבולית פעילה)"

    cum_fric_seg = np.cumsum(F_fric_seg)
    q_base = P_node[-1] / A
    return depths, w_node, P_node, t_mob_seg, F_fric_seg, cum_fric_seg, q_base, state_seg, stage_name

col_inputs, col_results = st.columns([1.25, 1.75])

with col_inputs:
    st.header("⚙️ 1. פרמטרים והפעלת עומס/שקיעה בראש")
    c1, c2 = st.columns(2)
    with c1:
        E = st.number_input("מודול אלסטיות חומר E (kPa)", value=30000000, step=1000000)
        D = st.number_input("קוטר הכלונס D (מטר)", min_value=0.2, max_value=3.0, value=0.6, step=0.05)
    with c2:
        B = st.number_input("מקדם בסיס פרבולי B (kN/m^0.5)", value=150.0, step=10.0)
        w0_mm = st.slider("תזוזה בראש הכלונס w₀ (מ\"מ) [מלמעלה למטה]", min_value=0.0, max_value=40.0, value=2.5, step=0.25)

    st.markdown("---")
    st.header("📂 2. נתוני השכבות (פרסיומטר וגזירה ישירה)")
    default_layers = pd.DataFrame({
        "עומק תחתית שכבה (מ')": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "לחץ פרסיומטר σ'h (kPa)": [90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 175.0, 190.0],
        "זווית חיכוך שיא δ_peak (°)": [32.0, 32.0, 32.0, 30.0, 30.0, 30.0, 33.0, 33.0, 34.0, 34.0],
        "תזוזת שיא w_peak (מ\"מ)": [2.0, 2.0, 2.5, 2.5, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0],
        "זווית משתיירת δ_res (°)": [26.0, 26.0, 26.0, 24.0, 24.0, 24.0, 27.0, 27.0, 28.0, 28.0],
        "תזוזת התייצבות w_res (מ\"מ)": [8.0, 8.0, 10.0, 10.0, 12.0, 12.0, 9.0, 9.0, 9.0, 9.0]
    })

    edited_df = st.data_editor(default_layers, num_rows="dynamic", use_container_width=True, hide_index=True)

layers_df = edited_df.dropna().sort_values(by="עומק תחתית שכבה (מ')").reset_index(drop=True)

if len(layers_df) > 0:
    depths, w_node, P_node, t_mob_seg, F_fric_seg, cum_fric_seg, q_base, state_seg, stage_name = shoot_top_down(
        w0_mm / 1000.0, layers_df, E, D, B
    )

    # חישוב עקומת עומס-שקיעה מלאה מ-0 (שלב א' + שלב ב')
    w0_sweep = np.linspace(0.0, 0.04, 65)
    P0_curve, w0_curve, Pbase_curve, cum_fric_curve = [], [], [], []
    for w_top in w0_sweep:
        _, wn, Pn, _, _, cf, _, _, _ = shoot_top_down(w_top, layers_df, E, D, B)
        w0_curve.append(wn * 1000.0)
        P0_curve.append(Pn)
        Pbase_curve.append(Pn[-1])
        cum_fric_curve.append(cf[-1])

    with col_results:
        st.info(f"**מצב נוכחי במודל:** {stage_name}")
        k1, k2, k3, k4 = st.columns(4)