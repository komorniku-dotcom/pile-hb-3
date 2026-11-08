import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import brentq

st.set_page_config(page_title="מודל כלונס מלמעלה למטה + חתך שכבות", layout="wide")
st.title("🏗️ מודל גיאומכני מלמעלה למטה – חתך שכבות ועקומת עומס-שקיעה (V8)")
st.markdown("---")

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

def shoot_top_down(w0, layers_df, E, D, B):
    A = np.pi * (D**2) / 4.0
    U = np.pi * D
    num_layers = len(layers_df)
    depths = [0.0] + layers_df["עומק תחתית שכבה (מ')"].tolist()

    w_node = np.zeros(num_layers + 1)
    P_node = np.zeros(num_layers + 1)
    t_mob_seg = np.zeros(num_layers)
    F_fric_seg = np.zeros(num_layers)
    state_seg = [""] * num_layers

    w_crit = np.zeros(num_layers + 1)
    P_crit = np.zeros(num_layers + 1)
    for i in range(num_layers - 1, -1, -1):
        dz = max(depths[i + 1] - depths[i], 0.01)
        sig_h = layers_df.loc[i, "לחץ פרסיומטר σ'h (kPa)"]
        t_p = sig_h * np.tan(np.radians(layers_df.loc[i, "זווית חיכוך שיא δ_peak (°)"]))
        t_r = sig_h * np.tan(np.radians(layers_df.loc[i, "זווית משתיירת δ_res (°)"]))
        w_p = layers_df.loc[i, "תזוזת שיא w_peak (מ\"מ)"] / 1000.0
        w_r = layers_df.loc[i, "תזוזת התייצבות w_res (מ\"מ)"] / 1000.0

        w_est = w_crit[i + 1] + (P_crit[i + 1] * dz) / (E * A) + 0.5 * w_p
        t_m, _ = calc_shear_stress(0.5 * (w_crit[i + 1] + w_est), t_p, t_r, w_p, w_r)
        F_f = U * dz * t_m
        P_crit[i] = P_crit[i + 1] + F_f
        w_crit[i] = w_crit[i + 1] + (0.5 * (P_crit[i] + P_crit[i + 1]) * dz) / (E * A)

    w0_limit = float(w_crit.item(0))

    if w0 <= w0_limit:
        def residual_stage_a(P0_guess):
            w_tmp = np.zeros(num_layers + 1)
            P_tmp = np.zeros(num_layers + 1)
            w_tmp[0] = w0
            P_tmp[0] = P0_guess
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
                    P_tmp[i + 1] = 0.0
                    w_tmp[i + 1] = 0.0
                else:
                    P_tmp[i + 1] = P_tmp[i] - F_f
                    P_mean = 0.5 * (P_tmp[i] + P_tmp[i + 1])
                    w_tmp[i + 1] = max(w_tmp[i] - (P_mean * dz) / (E * A), 0.0)
            return float(P_tmp[-1])

        try:
            P0_solved = brentq(residual_stage_a, 0.0, float(P_crit.item(0)) * 1.5)
        except Exception:
            P0_solved = float(P_crit.item(0)) * (w0 / max(w0_limit, 1e-6))

        w_node[0] = w0
        P_node[0] = P0_solved
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

        wb_solved = brentq(lambda wb: float(eval_from_wbase(wb)[0].item(0)) - w0, 0.0, 0.5)
        w_node, P_node, t_mob_seg, F_fric_seg, state_seg = eval_from_wbase(wb_solved)
        stage_name = "שלב ב': הקצה מגויס (תגובת בסיס פרבולית פעילה)"

    cum_fric_seg = np.cumsum(F_fric_seg)
    q_base = float(P_node[-1]) / A
    return depths, w_node, P_node, t_mob_seg, F_fric_seg, cum_fric_seg, q_base, state_seg, stage_name
col_inputs, col_results = st.columns([1.2, 1.8])

with col_inputs:
    st.header("⚙️ 1. פרמטרים והפעלת תזוזה בראש")
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

    w0_sweep = np.linspace(0.0, 0.04, 65)
    P0_curve, w0_curve, Pbase_curve, cum_fric_curve = [], [], [], []
    for w_top in w0_sweep:
        _, wn, Pn, _, _, cf, _, _, _ = shoot_top_down(float(w_top), layers_df, E, D, B)
        w0_curve.append(float(wn.item(0)) * 1000.0)
        P0_curve.append(float(Pn.item(0)))
        Pbase_curve.append(float(Pn[-1]))
        cum_fric_curve.append(float(cf[-1]))

    with col_results:
        st.info(f"**מצב נוכחי במודל:** {stage_name}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("עומס בראש P₀", f"{float(P_node.item(0)):,.1f} kN")
        k2.metric("תזוזה בראש w₀", f"{float(w_node.item(0))*1000:.2f} מ\"מ")
        k3.metric("עומס בקצה P_base", f"{float(P_node[-1]):,.1f} kN")
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
            file_name=f"pile_topdown_report_D{D}_w{w0_mm}mm.csv",
            mime="text/csv"
        )

        tab1, tab2 = st.tabs(["📊 חתך שכבות + עקומת עומס-שקיעה", "📋 טבלת התקדמות החזית לעומק"])

        with tab1:
            fig_main, (ax_soil, ax_ls) = plt.subplots(
                1, 2, figsize=(11, 5.2), gridspec_kw={'width_ratios': [1.0, 1.55]}
            )

            cmap = plt.get_cmap('Set3')
            max_depth = depths[-1]
            for i in range(len(layers_df)):
                z_top = depths[i]
                z_bot = depths[i + 1]
                dz = z_bot - z_top
                layer_color = cmap(i % 12)
                
                rect_soil = patches.Rectangle((-1.5, z_top), 3.0, dz, facecolor=layer_color, edgecolor='gray', alpha=0.75)
                ax_soil.add_patch(rect_soil)

                sig_h_val = layers_df.loc[i, "לחץ פרסיומטר σ'h (kPa)"]
                is_active = P_node[i] > 1e-4
                status_icon = "●" if is_active else "○"
                ax_soil.text(
                    0.42, z_top + dz / 2.0,
                    f"{status_icon} L{i+1}: σ'h={sig_h_val:.0f} kPa",
                    va='center', ha='left', fontsize=8, fontweight='bold' if is_active else 'normal',
                    color='#111111' if is_active else '#555555'
                )

            pile_width = 0.65
            rect_pile = patches.Rectangle(
                (-pile_width / 2.0, 0.0), pile_width, max_depth,
                facecolor='#4a5568', edgecolor='black', linewidth=1.8, zorder=4
            )
            ax_soil.add_patch(rect_pile)

            ax_soil.annotate(
                f"P₀={float(P_node.item(0)):.0f} kN",
                xy=(0, 0), xytext=(0, -max_depth * 0.11),
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1f497d',
                arrowprops=dict(facecolor='#1f497d', shrink=0.05, width=2.5, headwidth=8)
            )

            base_color = '#d9534f' if P_node[-1] > 1e-4 else '#a0aec0'
            ax_soil.plot([-pile_width/2.0, pile_width/2.0], [max_depth, max_depth], color=base_color, linewidth=4.5, zorder=5)
            ax_soil.text(
                0, max_depth + max_depth * 0.04,
                f"P_base={float(P_node[-1]):.0f} kN",
                ha='center', va='top', fontsize=8.5, fontweight='bold', color=base_color
            )

            ax_soil.set_xlim(-1.4, 1.45)
            ax_soil.set_ylim(max_depth * 1.12, -max_depth * 0.16)
            ax_soil.set_xticks([])
            ax_soil.set_ylabel("עומק z (מטר)")
            ax_soil.set_title("חתך הכלונס ושכבות הקרקע\n(● שכבה מגויסת | ○ נייחת)")

            ax_ls.plot(w0_curve, P0_curve, label='עומס כולל בראש הכלונס (P₀)', color='#1f497d', linewidth=2.5)
            ax_ls.plot(w0_curve, cum_fric_curve, '-.', label='חיכוך מעטפת מצטבר (ΣF)', color='green', linewidth=2)
            ax_ls.plot(w0_curve, Pbase_curve, '--', label='עומס שמגיע לקצה (P_base)', color='#d9534f', linewidth=2)
            ax_ls.scatter([w0_mm], [float(P_node.item(0))], color='black', s=85, zorder=5, label='נקודת הרצה נוכחית')
            ax_ls.set_xlabel("תזוזה בראש הכלונס w₀ (מ\"מ)")
            ax_ls.set_ylabel("כוח (kN)")
            ax_ls.set_title("עקומת עומס-שקיעה בראש הכלונס\n(משלב א' לשלב ב')")
            ax_ls.grid(True, linestyle='--', alpha=0.5)
            ax_ls.legend(fontsize=8.5)

            plt.tight_layout()
            st.pyplot(fig_main)

        with tab2:
            st.dataframe(results_df.style.format({
                "תזוזה w (מ\"מ)": "{:.2f}",
                "מאמץ בפועל t_mob (kPa)": "{:.1f}",
                "כוח שכבה (kN)": "{:.1f}",
                "חיכוך מצטבר ΣF (kN)": "{:,.1f}",
                "כוח צירי נותר P (kN)": "{:,.1f}"
            }), use_container_width=True, hide_index=True)

# END OF FILE
