import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import streamlit as st
from sklearn.neighbors import NearestNeighbors
import difflib
import math
from collections import OrderedDict

# --- CONFIG ---

# --- CONFIG: use paths relative to this file ---
from pathlib import Path

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR.parent / "data"

FEATURES_FILE = DATA_DIR / "polyeXplore model feature & index.xlsx"
LIBRARY_FILE  = DATA_DIR / "polyeXplore polymer library data.xlsx"
 
# --- DATA LOADING --- A

@st.cache_data
def load_data():
    features = pd.read_excel(FEATURES_FILE, sheet_name='polyFeature', index_col=0)
    properties = pd.read_excel(FEATURES_FILE, sheet_name='polyIndex', index_col=0)
    library = pd.read_excel(LIBRARY_FILE, sheet_name='reference', index_col=0)
    return features, properties, library

features, properties, library = load_data()

# --- PREPARE INDICES --- B

ppi = pd.DataFrame(features.values.dot(properties.values), index=features.index, columns=properties.columns)
ppi_mean = ppi.mean()
ppi_std = ppi.std(ddof=0)
ppi_z = (ppi - ppi_mean) / ppi_std

# --- STREAMLIT APP --- C

st.title("PolyeXplore : Polymer exploration model")
st.write("Welcome to Polymer eXploration Model!.")
st.markdown("© polyeXplore — Sibabrata De", unsafe_allow_html=True)

st.header("Polymer Structural Features vs. Polymer Properties Index")
st.markdown(
    """
    <div style='text-align: justify;'>
    <p>
    A novel model has been developed to visualize polymer properties based on the influence of key structural features
    of repeat unit in polymers. This simplified macro-level approach complements the fundamental 
    <i>'Group Contribution Adition'</i>, originally proposed by Herman F. Mark, which relates polymer properties to 
    their structural reapeat units. This concept was systematized by D.W. van Krevelen and later enhanced by 
    Jozef Bicerano into a comprehensive predictive framework  that underpins modern computational 
    polymer property prediction.
    </p>
    <p>
    In this model, the user provides a quantified input of the structural features of a polymer's repeat unit. 
    The model visualizes the weighted impact of these features on the polymer’s property index, which reflects 
    the property index hierarchy among commodity, engineering and high-performance polymers finally allowing comparison 
    against a reference polymer library. This enables insightful understanding of how polymer architecture 
    influences key polymer properties. 
    </p>
    </div>
    """,
    unsafe_allow_html=True
)



st.markdown("### 📊 Key Structural Features (SF) of Polymer Repeat Unit (RU)")
st.markdown(", ".join(features.columns))

st.markdown("### 📊 Polymer Properties influenced by SF's of polymer RU")
st.markdown(", ".join(properties.columns))

# --- USER INPUT --- D

if "polymer" not in st.session_state:
    st.session_state.polymer = ""
if "user_vals" not in st.session_state:
    st.session_state.user_vals = {c: 0.0 for c in features.columns}
if "input_saved" not in st.session_state:
    st.session_state.input_saved = False

st.markdown("### 📊 Structural Features of repeat units of polymers - computation of influence matrix")
st.markdown("Enter structural features of a polymer repeat unit to visualize its influence on property.")

valid_polymers = [str(p).upper() for p in features.index]
dataset_cols = list(features.columns)

def show_comparison():
    polymer = st.session_state.polymer
    user_vals = st.session_state.user_vals
    user_series = pd.Series(user_vals, name=f"{polymer}-user")
    orig_series = features.loc[polymer, dataset_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    orig_series.name = f"{polymer}-data"
    compare_df = pd.concat([orig_series, user_series], axis=1)
    st.subheader("📊 Feature scores — user input vs dataset")
    st.dataframe(compare_df)

    y = np.arange(len(compare_df))
    h = 0.35
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(y - h/2, compare_df.iloc[:, 0], h, label=compare_df.columns[0])
    ax.barh(y + h/2, compare_df.iloc[:, 1], h, label=compare_df.columns[1])
    ax.set_xlabel("Feature Value")
    ax.set_title(f"Structural Features: {polymer} — user vs dataset")
    ax.set_yticks(y)
    ax.set_yticklabels(compare_df.index)
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

# --- FORM INPUT --- E

if st.session_state.input_saved:
    st.success(f"✅ Inputs saved for **{st.session_state.polymer}**.")
    with st.expander("Show saved feature inputs"):
        st.dataframe(pd.Series(st.session_state.user_vals, name="Value"))
    cols = st.columns([1, 1])
    with cols:
        if st.button("Edit inputs"):
            st.session_state.input_saved = False
    with cols[1]:
        if st.button("Reset inputs"):
            st.session_state.polymer = ""
            st.session_state.user_vals = {c: 0.0 for c in features.columns}
            st.session_state.input_saved = False
    if st.session_state.input_saved and st.session_state.polymer:
        show_comparison()
else:
    with st.form("user_entry_form", clear_on_submit=False):
        raw_name = st.text_input("Enter the polymer name (use CAPITAL letters and numbers, example POM, PA66):", value=st.session_state.polymer).strip()
        cols_per_row = 4
        for i, fname in enumerate(dataset_cols):
            if i % cols_per_row == 0:
                row = st.columns(cols_per_row)
            col = row[i % cols_per_row]
            st.session_state.user_vals[fname] = col.number_input(
                label=fname,
                value=float(st.session_state.user_vals.get(fname, 0.0)),
                step=0.1,
                format="%.3f",
                key=f"feat_{fname}"
            )
        submitted = st.form_submit_button("Save")
        if submitted:
            polymer = raw_name.upper()
            if not polymer:
                st.error("Please enter a polymer name.")
            elif polymer not in valid_polymers:
                st.error("❌ Polymer not found. Please enter a valid polymer name.")
                sugg = difflib.get_close_matches(polymer, valid_polymers, n=5, cutoff=0.6)
                if sugg:
                    st.caption("Did you mean: " + ", ".join(sugg))
            else:
                st.session_state.polymer = polymer
                st.session_state.input_saved = True
                st.success(f"✅ Saved inputs for **{polymer}**. Continue to the sections below.")

# --- NEAREST POLYMERS --- F

st.markdown("### 📊 Nearest equivalent polymers based on structural features input")
if not st.session_state.get("input_saved"):
    st.info("Please complete and save the user entry above to compute nearest polymers.")
else:
    polymer = st.session_state.polymer
    feature_order = list(features.columns)
    user_feat = pd.Series(st.session_state.user_vals, dtype=float).reindex(feature_order)
    user_ppi = user_feat @ properties
    user_ppi.name = f"{polymer}-user"
    user_ppi_z = (user_ppi - ppi_mean) / ppi_std

    props_for_nn = list(ppi_z.columns)
    X = ppi_z[props_for_nn].astype(float).values
    y = user_ppi_z[props_for_nn].astype(float).values.reshape(1, -1)

    nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(y)
    neighbor_idx = indices[0]
    neighbor_dist = distances
    neighbor_names = ppi_z.index[neighbor_idx].tolist()

    neighbor_idx = indices[0]
    neighbor_dist = distances[0]
    neighbor_names = ppi_z.index[neighbor_idx].tolist()

    n_neighbors_found = len(neighbor_names)

    order = np.argsort(neighbor_dist)
    order = order[:n_neighbors_found]  # safely limit order to existing neighbors

    names_ord = [neighbor_names[i] for i in order if i < n_neighbors_found]
    dists_ord = np.array(neighbor_dist)[order]

    # Now safe to compute relative distances
    min_dist = np.min(dists_ord)
    if min_dist <= 0:
        min_dist = 1e-6

    rel_dist = [(max(0, (d - min_dist) / min_dist)) for d in dists_ord]
    rel_dist = [round(d, 3) for d in rel_dist]

# Continue plotting...

    st.subheader("1. Nearest equivalent polymers based on input features")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names_ord[::-1], rel_dist[::-1])
    ax.set_xlabel("Relative distance in PPI space (min = 0)")
    ax.set_ylabel("Polymer")
    ax.set_title("Nearest polymers)")
    for y, v in enumerate(rel_dist[::-1]):
        ax.text(v, y, f" {v:.2f}", va="center")
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("2. Properties of nearest polymers from reference library")

    # 1. Get property values from reference library for the nearest polymers
    neighbor_library = library.loc[names_ord]  # DataFrame: index=polymer, columns=properties

    # 2. Transpose so polymer names are columns, properties are rows (like your image)
    neighbor_library_T = neighbor_library.T

    # 3. Display in Streamlit with formatting (optional: 3 decimals)
    st.dataframe(neighbor_library_T.style.format("{:.2f}"))

    st.markdown("Note: The nearest polymers are determined based on Euclidean distance in the standardized property index space.")

# --- END OF USER INPUT SECTION ---

# ===== Category-based polymer selection =====

st.markdown("### 📊 Polymer property hierarchy by category for visualization")

# Define your categories in desired order
categories_ordered = [
    ("Commodity", [
        "PE", "PP", "PS", "PVC", "ABS", "PMMA", "ASA"
    ]),
    ("Engineering", [
        "PA12", "PA6", "PA66", "PBT", "PET", "POM-H", "POM-C",
        "POK", "COC", "PC", "PAR", "PPE", "PA46", "PPS", "PARA"
    ]),
    ("High-performance", [
        "PPA", "PEI", "PES", "PSU", "PAI", "PI", "LCP", "PEEK", "PEK", "PBI"
    ]),
]

# Only keep polymers that exist in your data
available = set(ppi_z.index.tolist())
categories = [(name, [p for p in plist if p in available]) for name, plist in categories_ordered]

# Session state defaults (so select/clear buttons can control multiselects)
for cat_name, _ in categories:
    key = f"sel_{cat_name.replace('-', '_').replace(' ', '_').lower()}"
    if key not in st.session_state:
        st.session_state[key] = []

# Top-level buttons (optional)
c0, c1 = st.columns(2)
if c0.button("Select ALL (across categories)"):
    for cat_name, plist in categories:
        st.session_state[f"sel_{cat_name.replace('-', '_').replace(' ', '_').lower()}"] = plist[:]
if c1.button("Clear ALL"):
    for cat_name, _ in categories:
        st.session_state[f"sel_{cat_name.replace('-', '_').replace(' ', '_').lower()}"] = []

# Render three columns for three categories
colA, colB, colC = st.columns(3)
cols = [colA, colB, colC]

selected_lists = []
for (cat_name, plist), col in zip(categories, cols):
    key = f"sel_{cat_name.replace('-', '_').replace(' ', '_').lower()}"
    with col:
        st.subheader(cat_name)
        b1, b2 = st.columns(2)
        if b1.button("All", key=f"all_{key}"):
            st.session_state[key] = plist[:]
        if b2.button("Clear", key=f"clr_{key}"):
            st.session_state[key] = []
        selected = st.multiselect(
            f"Select {cat_name.lower()} polymers",
            options=plist,
            default=st.session_state[key],
            key=key
        )
        selected_lists.append(selected)

# Combine selections in category order (preserve order, remove dups)

from collections import OrderedDict
sel_polymers = list(OrderedDict.fromkeys([p for sub in selected_lists for p in sub]))

# ===== Property selection (limit to 8, with quick buttons) =====
st.markdown("### 📊 Pick indicative properties for visualization of polymer property rankings")
prop_options = ppi_z.columns.tolist()
prop_options = prop_options[:8]  # if you truly want to cap to first 8; remove this line if not

if "sel_props" not in st.session_state:
    st.session_state.sel_props = prop_options[:]

p1, p2 = st.columns(2)
if p1.button("All properties"):
    st.session_state.sel_props = prop_options[:]
if p2.button("Clear properties"):
    st.session_state.sel_props = []

sel_props = st.multiselect(
    "Select properties:",
    options=prop_options,
    default=st.session_state.sel_props,
    key="sel_props"
)

# ===== Show results =====
if sel_polymers and sel_props:
    subset_ppi = ppi_z.loc[sel_polymers, sel_props]

    st.markdown("### 📊 Heatmap")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(subset_ppi, annot=True, cmap="RdYlBu", center=0,
                annot_kws={"size": 8, "color": "grey"}, ax=ax)
    ax.set_xlabel("Property")
    ax.set_ylabel("Polymer")
    ax.set_title("Polymer Property Index — category selection")
    fig.tight_layout()
    st.pyplot(fig)
else:
    st.info("Pick at least one polymer (by category) and one property to display the values.")


# Randomly pick 5 polymers (rows) for showing property index rankings

st.markdown("### 📊 A random selection of polymers and property to display rankings")

sampled_polymers = ppi_z.sample(n=5, random_state=42)  # random_state for reproducibility

# Select 2 properties (columns)
selected_properties = np.random.choice(ppi_z.columns, size=3, replace=False)
subset_ppi = sampled_polymers[selected_properties]

# subset_ppi = ppi_z.loc[sampled_polymers.index, ["Heat Deflection", "Crystallinity"]]

st.markdown(f"Polymer Property Index (random selection of 5 polymers and {len(selected_properties)} properties):")

# Plot heatmap
fig, ax = plt.subplots(figsize=(8, 5))   # create figure & axis
sns.heatmap(subset_ppi,
            annot=True,
            cmap="RdYlBu",
            center=0,
            annot_kws={"size": 8, "color": "grey"},
            ax=ax)   # important: pass ax here

ax.set_title("Polymer Property Index (random selection)")
ax.set_xlabel("Property")
ax.set_ylabel("Polymer")

st.pyplot(fig)   # render inside Streamlit app

# # Polymer Property vs Structural Feature Correaltion

st.markdown("### 📊 Polymers - Property vs Structural Feature Correlation  ")

corr_matrix = pd.DataFrame({
    prop: [features[f].corr(ppi[prop]) for f in features.columns]
    for prop in ppi.columns
}, index=features.columns)

# Heatmap

# st.markdown("Structural Feature  <--> Polymer Property Correlations")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix,
            annot=True,
            cmap="RdYlBu",
            center=0,
            ax=ax)

ax.set_title("Feature ↔ Property correlations of polymers")
st.pyplot(fig)

# Contribution map #

# --- Influence sign map (Positive / Negative / Neutral) ---
# st.markdown("### ➕/➖ Influence Sign by Feature → Property")
# sign_df = corr_matrix.applymap(lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))
# st.dataframe(sign_df)

# Create a sign matrix with categorical labels
# Existing sign matrix with categorical labels

st.markdown("### 📊 Structural Feature(SF) - Polymer Property Index(PPI) : Influence Matrix")

sign_df = corr_matrix.applymap(lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))

# Define a color mapping for the categories 
color_map = {
    "Positive": "background-color: #e6550d; color: white;",  # orange/red with white text
    "Negative": "background-color: #3182bd; color: white;",  # blue with white text
    "Neutral": "background-color: #f0f0f0; color: black;"    # light grey with black text
}

def style_cells(val):
    return color_map.get(val, "")

# Apply styling to the DataFrame
styled_sign_df = sign_df.style.applymap(style_cells)

# Display styled DataFrame in Streamlit
st.dataframe(styled_sign_df)



# --- Top 3 features per property (both directions) ---
st.markdown("### 📊 SF - PPI Influence matrix")

rows = []
for prop in corr_matrix.columns:
    s = corr_matrix[prop].dropna()
    top_pos = s.sort_values(ascending=False).head(3)
    top_neg = s.sort_values(ascending=True).head(3)

    for feat, val in top_pos.items():
        rows.append({"Property": prop, "Feature": feat, "Correlation": float(val), "Direction": "Positive"})
    for feat, val in top_neg.items():
        rows.append({"Property": prop, "Feature": feat, "Correlation": float(val), "Direction": "Negative"})

top3_df = pd.DataFrame(rows)
st.dataframe(top3_df)

# --- Optional: inspect a single property with a bar chart ---
# st.markdown("### 🔍 Inspect a property")
# prop_choice = st.selectbox("Pick a property to inspect:", corr_matrix.columns.tolist())
# k = 3  # top-k
# s = corr_matrix[prop_choice].dropna()
# top_pos = s.sort_values(ascending=False).head(k)
# top_neg = s.sort_values(ascending=True).head(k)

# fig, ax = plt.subplots(figsize=(7, 4))
# plot_s = pd.concat([top_pos, top_neg]).sort_values()
# ax.barh(plot_s.index, plot_s.values)
# ax.axvline(0, linewidth=1)
# ax.set_title(f"Top ±{k} features for {prop_choice}")
# ax.set_xlabel("Correlation")
# fig.tight_layout()
# st.pyplot(fig)

# --- Bar charts for positive/negative influence on each property ---
# --- Cumulative Positive/Negative Contribution per Property ---
st.markdown("### 📊 Positive vs Negative contribution of SF's on PPI")

# Split correlations into positive and negative per property
pos_contrib = corr_matrix.clip(lower=0).sum(axis=0)
neg_contrib = corr_matrix.clip(upper=0).abs().sum(axis=0)

# Normalize so positive + negative = 100%
total = pos_contrib + neg_contrib
pos_pct = (pos_contrib / total * 100).fillna(0)
neg_pct = (neg_contrib / total * 100).fillna(0)

# Combine into one DataFrame
posneg_df = pd.DataFrame({
    "Positive": pos_pct,
    "Negative": neg_pct
}).loc[corr_matrix.columns[:8]]   # limit to 8 properties

# Plot stacked horizontal bar chart
fig, ax = plt.subplots(figsize=(8, 5))
posneg_df.plot(kind="bar", stacked=True, color=["Red", "Yellow"], ax=ax)

ax.set_ylabel("Properties")
ax.set_xlabel("Contribution (%)")
ax.set_title("Relative Positive vs Negative contribution of SF's on Property")
ax.legend(loc="upper right")
fig.tight_layout()

st.pyplot(fig)
# ===================== end section =====================

import math

# Cumulative positive/negative contribution of a fixed set of features (9) on a fixed set of properties (8)

# 2) Select 9 features and 8 properties
features_9  = list(features.columns)[:9]
properties_8 = list(corr_matrix.columns)[:8]
cm = corr_matrix.loc[features_9, properties_8]

# 3) Aggregate positive and negative contributions per property
pos_sum = cm.clip(lower=0).sum(axis=0)   # sum of positive correlations per property
neg_sum = cm.clip(upper=0).abs().sum(axis=0)  # sum of negative correlations per property

# 4) Put into a DataFrame for plotting
summary_df = pd.DataFrame({
    "Positive": pos_sum,
    "Negative": neg_sum
}, index=properties_8)

# -----------------------------------------
# End of script
# -----------------------------------------


