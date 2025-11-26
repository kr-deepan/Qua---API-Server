import numpy as np
import joblib
import bentoml
# Note: bentoml.io.JSON is deprecated; we use standard Python types now

# ---------------------------------------------------------
# Model paths (relative to this file)
# ---------------------------------------------------------
PRIMARY_MODEL_PATH = "models/primary_full_model.pkl"
BIO_MODEL_PATH = "models/biological_full_model.pkl"
TERTIARY_MODEL_PATH = "models/tertiary_full_model.pkl"

# ---------------------------------------------------------
# Feature & target columns (Same as before)
# ---------------------------------------------------------
PRIMARY_FEATURE_COLS = [
    "Q_in_mld", "temp_C", "pH", "TSS_in_mgL", "BOD5_in_mgL", "COD_in_mgL",
    "oil_grease_in_mgL", "peak_factor", "screen_type", "bar_spacing_mm", "approach_velocity_ms",
    "screen_angle_deg", "open_area_fraction", "num_screens", "grit_type", "chamber_length_m", "chamber_width_m",
    "water_depth_m", "inlet_velocity_ms", "detention_time_s", "air_flow_m3h_per_m", "clarifier_type", "tank_surface_area_m2", "side_water_depth_m",
    "weir_length_m", "sludge_withdrawal_rate_m3h", "HRT_h", "surface_loading_rate_m3m2h", "weir_loading_m3mh", "recycle_ratio_pct", "air_release_pressure_bar",
    "saturator_retention_time_min", "coagulant_dose_mgL", "polymer_dose_mgL", "bubble_diameter_um",
]

PRIMARY_TARGET_COLS = [
    "screen_TSS_removal_eff_pct", "TSS_after_screen_mgL", "Q_after_screen_mld", "screen_energy_kwh_day",
    "grit_removal_eff_pct", "TSS_after_grit_mgL", "Q_after_grit_mld", "grit_energy_kwh_day",
    "sed_TSS_removal_eff_pct", "sed_BOD_removal_eff_pct", "sed_COD_removal_eff_pct", "TSS_after_sed_mgL",
    "BOD5_after_sed_mgL", "COD_after_sed_mgL", "sed_energy_kwh_day", "daf_TSS_removal_eff_pct",
    "daf_OG_removal_eff_pct", "daf_COD_removal_eff_pct", "TSS_final_mgL", "BOD5_final_mgL",
    "COD_final_mgL", "oil_grease_final_mgL", "Q_final_mld", "daf_energy_kwh_day",
]

BIO_FEATURE_COLS = [
    "Q_bio_mld", "temp_C", "pH", "TSS_in_bio_mgL", "BOD5_in_bio_mgL", "COD_in_bio_mgL",
    "NH4_in_mgL", "NO3_in_mgL", "MLSS_mgL", "SRT_days", "aeration_rate_m3min",
    "DO_mgL", "HRT_AS_h", "F_M_ratio_kgkgd", "filter_depth_m", "air_flow_m3m2min",
    "HRT_bio_h", "biofilter_surface_area_m2",
]

BIO_TARGET_COLS = [
    "BOD_eff_AS_pct", "COD_eff_AS_pct", "NH4_eff_AS_pct", "BOD_after_AS_mgL", "COD_after_AS_mgL",
    "NH4_after_AS_mgL", "ASP_energy_kwh_day", "NH4_eff_bio_pct", "BOD_polish_eff_pct", "COD_polish_eff_pct",
    "NO3_reduction_eff_pct", "BOD_final_bio_mgL", "COD_final_bio_mgL", "NH4_final_mgL", "NO3_final_mgL",
    "biofilter_energy_kwh_day", "BOD_total_eff_pct", "COD_total_eff_pct", "NH4_total_eff_pct", "system_efficiency_pct",
    "oxygen_utilization_pct", "total_bio_energy_kwh_day",
]

TER_FEATURE_COLS = [
    "Q_ter_mld", "temp_C", "pH_bulk", "TSS_after_bio_mgL", "turbidity_in_NTU", "BOD_in_ter_mgL", "COD_in_ter_mgL",
    "NH4_in_ter_mgL", "NO3_in_ter_mgL", "TP_in_ter_mgL", "Ecoli_in_CFU_100mL", "micropollutant_in_ugL",
    "membrane_pore_size_um", "filtration_pressure_bar", "membrane_area_m2", "flux_LMH",
    "DO_ter_mgL", "SRT_ter_days", "pH_ter", "carbon_dose_mgL", "mixing_speed_rpm",
    "anoxic_time_min", "coagulant_dose_mgL", "mixing_intensity_s_1", "UV_dose_mJcm2", "lamp_power_W",
    "UV_contact_time_s", "UVT_pct", "ozone_dose_mgL", "H2O2_mgL",
    "temp_AOP_C", "contact_time_AOP_min",
]

TER_TARGET_COLS = [
    "turbidity_removal_membrane_pct", "TSS_removal_membrane_pct", "TSS_after_membrane_mgL", "turbidity_after_membrane_NTU",
    "pathogen_LRV_membrane", "Ecoli_after_membrane_CFU_100mL", "membrane_energy_kwh_day", "BOD_after_membrane_mgL",
    "COD_after_membrane_mgL", "micropollutant_after_membrane_ugL", "TN_polish_eff_pct", "TP_polish_eff_pct",
    "TN_after_nutrient_mgL", "TP_after_nutrient_mgL", "BOD_after_nutrient_mgL", "COD_after_nutrient_mgL",
    "nutrient_energy_kwh_day", "LRV_uv", "Ecoli_after_uv_CFU_100mL", "pathogen_removal_uv_pct",
    "UV_energy_kwh_day", "micropollutant_removal_AOP_pct", "micropollutant_final_ugL", "COD_AOP_eff_pct",
    "COD_final_ter_mgL", "BOD_final_ter_mgL", "AOP_energy_kwh_day", "turbidity_final_NTU", "TN_final_mgL",
    "TP_final_mgL", "NH4_final_ter_mgL", "NO3_final_ter_mgL", "Ecoli_final_CFU_100mL", "TOC_final_mgL",
    "overall_pathogen_removal_pct", "tertiary_total_energy_kwh_day", "meets_agricultural_reuse",
    "meets_industrial_reuse", "meets_potable_reuse",
]

# ---------------------------------------------------------
# Helpers & Recommendations (KEPT AS IS)
# ---------------------------------------------------------

def energy_per_m3(energy_kwh_day: float, Q_mld: float):
    Q_m3_day = Q_mld * 1000.0
    if Q_m3_day <= 0: return None
    return energy_kwh_day / Q_m3_day

def _build_feature_array(payload: dict, feature_cols):
    # Added safe get() to prevent crashes if a key is missing
    return np.array([[payload.get(col, 0.0) for col in feature_cols]], dtype=float)

def recommend_screening(inp: dict, outp: dict):
    recs = []
    eff = outp["screen_TSS_removal_eff_pct"]
    bar_spacing = inp["bar_spacing_mm"]
    velocity = inp["approach_velocity_ms"]
    open_area = inp["open_area_fraction"]
    Q = inp["Q_in_mld"]
    e_kwh_day = outp["screen_energy_kwh_day"]
    epm3 = energy_per_m3(e_kwh_day, Q)

    if eff < 25:
        if bar_spacing > 15: recs.append(f"Screening efficiency is low ({eff:.1f}%). Reduce bar spacing from {bar_spacing:.1f} mm towards 10–15 mm.")
        if velocity < 0.7 or velocity > 1.0: recs.append(f"Approach velocity is {velocity:.2f} m/s. Adjust to 0.7–0.9 m/s for optimal capture.")
        if open_area < 0.5: recs.append(f"Open area fraction is {open_area:.2f}. Consider screens with ≥0.5 open area.")
    if epm3 and epm3 > 0.015: recs.append(f"Screening specific energy is about {epm3*1000:.1f} Wh/m³. Optimize screen runtime.")
    if not recs: recs.append(f"Screening is operating efficiently ({eff:.1f}% TSS removal). Maintain current settings.")
    return recs

def recommend_grit(inp: dict, outp: dict):
    recs = []
    eff = outp["grit_removal_eff_pct"]
    v = inp["inlet_velocity_ms"]
    dt = inp["detention_time_s"]
    grit_type = "aerated" if inp["grit_type"] == 1 else "plain"
    if eff < 60:
        recs.append(f"Grit removal efficiency is {eff:.1f}%. Increase detention time towards 60–80 s.")
        if dt < 50: recs.append(f"Current detention time is {dt:.0f} s. Increase tank volume.")
        if v > 0.35: recs.append(f"Inlet velocity {v:.2f} m/s is high. Adjust to < 0.30 m/s.")
    if grit_type == "aerated":
        e = outp["grit_energy_kwh_day"]
        if e > 10: recs.append(f"Aerated grit energy {e:.1f} kWh/d. Optimize blower.")
    if not recs: recs.append(f"Grit removal is stable ({eff:.1f}%). Maintain current operation.")
    return recs

def recommend_sedimentation(inp: dict, outp: dict):
    recs = []
    tss_eff = outp["sed_TSS_removal_eff_pct"]
    HRT = inp["HRT_h"]
    SLR = inp["surface_loading_rate_m3m2h"]
    weir_load = inp["weir_loading_m3mh"]
    if tss_eff < 50: recs.append(f"Primary clarifier TSS removal ({tss_eff:.1f}%) is low. Increase HRT or reduce SLR.")
    if HRT < 1.5: recs.append(f"HRT is {HRT:.2f} h. Increase volume.")
    if SLR > 3.0: recs.append(f"SLR is {SLR:.2f} m³/m²·h (high).")
    if weir_load > 20: recs.append(f"Weir loading {weir_load:.1f} m³/m·h is high.")
    if not recs: recs.append(f"Primary sedimentation is performing well ({tss_eff:.1f}% removal). Maintain current settings.")
    return recs

def recommend_daf(inp: dict, outp: dict):
    recs = []
    tss_eff = outp["daf_TSS_removal_eff_pct"]
    og_eff = outp["daf_OG_removal_eff_pct"]
    recycle = inp["recycle_ratio_pct"]
    coagulant = inp["coagulant_dose_mgL"]
    bubbles = inp["bubble_diameter_um"]
    Q = inp["Q_in_mld"]
    e = outp["daf_energy_kwh_day"]
    epm3 = energy_per_m3(e, Q)
    if tss_eff < 70 or og_eff < 80:
        recs.append(f"DAF removal modest. Increase coagulant/recycle.")
        if coagulant < 30: recs.append(f"Coagulant {coagulant:.1f} mg/L is low.")
        if recycle < 10: recs.append(f"Recycle {recycle:.1f}% is low.")
        if bubbles > 100: recs.append(f"Bubbles {bubbles:.0f} µm are large.")
    if epm3 and epm3 > 0.05: recs.append(f"DAF specific energy high ({epm3*1000:.1f} Wh/m³).")
    if not recs: recs.append(f"DAF is achieving high removals. Maintain current settings.")
    return recs

def generate_recommendations_primary(inputs: dict, outputs: dict):
    return { "screening": recommend_screening(inputs, outputs), "grit": recommend_grit(inputs, outputs), "sedimentation": recommend_sedimentation(inputs, outputs), "daf": recommend_daf(inputs, outputs) }

def recommend_asp(inp: dict, outp: dict):
    recs = []
    bod_eff = outp["BOD_eff_AS_pct"]
    nh4_eff = outp["NH4_eff_AS_pct"]
    MLSS = inp["MLSS_mgL"]
    DO = inp["DO_mgL"]
    FM = inp["F_M_ratio_kgkgd"]
    if bod_eff < 85: recs.append(f"ASP BOD removal {bod_eff:.1f}% low. Increase SRT.")
    if nh4_eff < 40: recs.append(f"Nitrification low ({nh4_eff:.1f}%). Keep DO > 2 mg/L.")
    if DO < 1.5: recs.append(f"DO {DO:.2f} mg/L is low.")
    if FM > 0.5: recs.append(f"F/M {FM:.2f} is high.")
    if not recs: recs.append(f"ASP is performing well. Maintain current operation.")
    return recs

def recommend_biofilter(inp: dict, outp: dict):
    recs = []
    nh4_eff = outp["NH4_eff_bio_pct"]
    BOD_polish = outp["BOD_polish_eff_pct"]
    HRT = inp["HRT_bio_h"]
    air = inp["air_flow_m3m2min"]
    if nh4_eff < 70: recs.append(f"Biofilter NH4 removal {nh4_eff:.1f}% low.")
    if HRT < 3: recs.append(f"Biofilter HRT {HRT:.1f} h low.")
    if air < 1.0: recs.append(f"Air flow {air:.2f} low.")
    if BOD_polish < 10: recs.append(f"BOD polish {BOD_polish:.1f}% low.")
    if not recs: recs.append(f"Biofilter is providing good polishing. Maintain settings.")
    return recs

def recommend_bio_overall(outp: dict):
    recs = []
    BOD_tot = outp["BOD_total_eff_pct"]
    NH4_tot = outp["NH4_total_eff_pct"]
    oxy = outp["oxygen_utilization_pct"]
    energy = outp["total_bio_energy_kwh_day"]
    if BOD_tot < 90: recs.append(f"Overall BOD removal {BOD_tot:.1f}% low.")
    if NH4_tot < 80: recs.append(f"Total NH4 removal {NH4_tot:.1f}% low.")
    if oxy < 70: recs.append(f"Oxygen utilization {oxy:.1f}% low.")
    if energy > 200: recs.append(f"Bio energy {energy:.1f} kWh/d high.")
    if not recs: recs.append(f"Biological system is performing well.")
    return recs

def generate_recommendations_biological(inputs: dict, outputs: dict):
    return { "asp": recommend_asp(inputs, outputs), "biofilter": recommend_biofilter(inputs, outputs), "overall": recommend_bio_overall(outputs) }

def recommend_membrane(inp: dict, outp: dict):
    recs = []
    turb_out = outp["turbidity_after_membrane_NTU"]
    tss_out = outp["TSS_after_membrane_mgL"]
    pore = inp["membrane_pore_size_um"]
    flux = inp["flux_LMH"]
    if turb_out > 1.0 or tss_out > 5:
        recs.append(f"Membrane effluent poor. Check pore/flux.")
        if pore > 0.2: recs.append(f"Pore {pore:.2f} µm large.")
        if flux > 80: recs.append(f"Flux {flux:.1f} LMH high.")
    if not recs: recs.append(f"Membrane achieving low turbidity/TSS. Maintain.")
    return recs

def recommend_uv(inp: dict, outp: dict):
    recs = []
    LRV = outp["LRV_uv"]
    ecoli = outp["Ecoli_final_CFU_100mL"]
    dose = inp["UV_dose_mJcm2"]
    UVT = inp["UVT_pct"]
    if LRV < 3 or ecoli > 10:
        recs.append(f"UV disinfection low. Increase dose.")
        if dose < 30: recs.append(f"Dose {dose:.1f} mJ/cm² low.")
        if UVT < 85: recs.append(f"UVT {UVT:.1f}% low.")
    if not recs: recs.append(f"UV providing strong disinfection. Maintain.")
    return recs

def recommend_aop(inp: dict, outp: dict):
    recs = []
    micro_final = outp["micropollutant_final_ugL"]
    oz = inp["ozone_dose_mgL"]
    h2o2 = inp["H2O2_mgL"]
    if micro_final > 0.1:
        recs.append(f"Micropollutant residual {micro_final:.3f} µg/L high.")
        if oz < 5: recs.append(f"Ozone {oz:.1f} mg/L low.")
        if h2o2 < 50: recs.append(f"H2O2 {h2o2:.1f} mg/L low.")
    if not recs: recs.append(f"AOP achieving good micropollutant removal. Maintain.")
    return recs

def generate_recommendations_tertiary(inputs: dict, outputs: dict):
    return { "membrane": recommend_membrane(inputs, outputs), "uv": recommend_uv(inputs, outputs), "aop": recommend_aop(inputs, outputs) }


# ---------------------------------------------------------
# NEW BENTOML (v1.2+) SERVICE DEFINITION
# ---------------------------------------------------------

@bentoml.service(name="aquasmart_service")
class AquaSmartService:
    def __init__(self):
        # Load models ONCE when the server starts
        self.primary_model = joblib.load(PRIMARY_MODEL_PATH)
        self.biological_model = joblib.load(BIO_MODEL_PATH)
        self.tertiary_model = joblib.load(TERTIARY_MODEL_PATH)

    @bentoml.api
    def primary(self, request_json: dict) -> dict:
        x = _build_feature_array(request_json, PRIMARY_FEATURE_COLS)
        y_pred = self.primary_model.predict(x)[0]
        outputs = dict(zip(PRIMARY_TARGET_COLS, y_pred))
        recs = generate_recommendations_primary(request_json, outputs)
        return { "outputs": outputs, "recommendations": recs }

    @bentoml.api
    def biological(self, request_json: dict) -> dict:
        x = _build_feature_array(request_json, BIO_FEATURE_COLS)
        y_pred = self.biological_model.predict(x)[0]
        outputs = dict(zip(BIO_TARGET_COLS, y_pred))
        recs = generate_recommendations_biological(request_json, outputs)
        return { "outputs": outputs, "recommendations": recs }

    @bentoml.api
    def tertiary(self, request_json: dict) -> dict:
        x = _build_feature_array(request_json, TER_FEATURE_COLS)
        y_pred = self.tertiary_model.predict(x)[0]
        outputs = dict(zip(TER_TARGET_COLS, y_pred))
        recs = generate_recommendations_tertiary(request_json, outputs)
        return { "outputs": outputs, "recommendations": recs }