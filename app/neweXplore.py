# General imports
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
    RDKit_OK = True
except Exception:
    RDKit_OK = False

# One-time heads-up if RDKit is unavailable
if not RDKit_OK:
    st.info(
        "RDKit is not available on this deployment. "
        "Descriptor fields will be shown as blank."
    )
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.neighbors import NearestNeighbors
import difflib
import math
from collections import OrderedDict

# --- rdkit & py3dmol functions ---
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, Crippen
import py3Dmol

# --- CONFIG ---
DATA_DIR = 'E:/15 Polyexplore/predictions_polyprop/'
FEATURES_FILE = DATA_DIR + 'polyeXplore model feature & index.xlsx'
LIBRARY_FILE = DATA_DIR + 'polyeXplore polymer library data.xlsx'
SMILES_FILE = DATA_DIR + 'polyeXplore_smiles.xlsx'


# --- DATA LOADING ---

@st.cache_data
def load_data():
    features = pd.read_excel(FEATURES_FILE, sheet_name='polyFeature', index_col=0)
    properties = pd.read_excel(FEATURES_FILE, sheet_name='polyIndex', index_col=0)
    library = pd.read_excel(LIBRARY_FILE, sheet_name='reference', index_col=0)
    return features, properties, library

features, properties, library = load_data()


# Load SMILES dataframe once
@st.cache_data
def load_smiles(file):
    df = pd.read_excel(file, sheet_name='SMILES')
    df.set_index('Polymer', inplace=True)
    return df

smiles_df = load_smiles(SMILES_FILE)

# --- PREPARE INDICES ---

ppi = pd.DataFrame(features.values.dot(properties.values),
                   index=features.index,
                   columns=properties.columns)
ppi_mean = ppi.mean()
ppi_std = ppi.std(ddof=0)
ppi_z = (ppi - ppi_mean) / ppi_std

# --- STREAMLIT APP Title, copyright info ---


st.title("PolyeXplore : Polymer exploration model")
st.write("'It is the lone worker who makes the first advance in a subject, the details may be worked out by a team' - Alexander Fleming - 1881–1955")
st.write("Welcome to Polymer eXploration Model!.")
# st.markdown("© polyeXplore — Sibabrata De", unsafe_allow_html=True)

st.header("Polymer structural features vs. polymer properties")

st.markdown("### Introduction")

st.markdown(
    """
    <div style='text-align: justify;'>
    <p>
    A novel model has been developed to visualize polymer properties based on the influence of key structural features
    of repeat units in polymers. This simplified macro-level approach complements the fundamental 
    <i>'Group Contribution Addition'</i>, originally proposed by Herman F. Mark, which relates polymer properties to 
    their structural repeat units. This concept was systematized by D.W. van Krevelen and later enhanced by 
    Jozef Bicerano into a comprehensive predictive framework that underpins modern computational 
    polymer property prediction.
    </p>
    <p>
    In this model, the user provides a quantified input of the structural features of a polymer's repeat unit. 
    The model visualizes the weighted impact of these features on the polymer’s property index, which reflects 
    the property index hierarchy among commodity, engineering, and high-performance polymers finally allowing comparison 
    against a reference polymer library. This enables insightful understanding of how polymer architecture 
    influences key polymer properties. 
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- DATA OVERVIEW ---

st.markdown("### 📊 Key structural features (SF) of polymer repeat Unit (RU)")
st.markdown(", ".join(features.columns))

st.markdown("### 📊 Polymer properties influenced by SF's of polymer RU")
st.markdown(", ".join(properties.columns))

# --- USER FEATURE INPUT ---

if "polymer" not in st.session_state:
    st.session_state.polymer = ""
if "user_vals" not in st.session_state:
    st.session_state.user_vals = {c: 0.0 for c in features.columns}
if "input_saved" not in st.session_state:
    st.session_state.input_saved = False

st.markdown("### 📊 Structural features of repeat units of polymers - exploration of influence vector")
st.markdown("Enter structural features of a polymer repeat unit to visualize its influence on property.")

valid_polymers = [str(p).upper() for p in features.index]
dataset_cols = list(features.columns)


# rdkit strat
# --- Polymer name input and 3D visualization (before form) ---

polymer_name_input = st.text_input(
    "Enter the polymer name (use CAPITAL letters and numbers, example POM, PA66):", 
    value=st.session_state.polymer
).strip()

if polymer_name_input:
    if polymer_name_input in smiles_df.index:
        smiles = smiles_df.loc[polymer_name_input, "SMILE_repeating_unit"]
        mol = Chem.MolFromSmiles(smiles)
        
        if mol:
            mol_3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(mol_3d)
            

            mol_block = Chem.MolToMolBlock(mol_3d)
            view = py3Dmol.view(width=600, height=400)
            view.addModel(mol_block, 'mol')
            view.setStyle({'stick': {}})
            view.zoomTo()
            view.show()
            # Add border around viewer
            view_html = view._make_html()
            bordered_html = f'''
            <div style="border: 2px solid #888; border-radius: 10px; padding: 8px; width: 620px; margin: auto;">
            {view_html}
            </div>
            '''
            st.components.v1.html(bordered_html, height=450)

            # Display SMILES and description if available

            if "Description" in smiles_df.columns:
                desc = smiles_df.loc[polymer_name_input, "Description"]
                if desc and str(desc).strip():
                    st.markdown(f"**Polymer Description:** {desc}")
            
            # (NEW) ---- DESCRIPTOR CALCULATION & DISPLAY SECTION -----
            def calculate_properties(smiles):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        return [None]*8  # Always 8 fields!
                    molwt = Descriptors.MolWt(mol)
                    tpsa = rdMolDescriptors.CalcTPSA(mol)
                    logp = Crippen.MolLogP(mol)
                    rot_bonds = Lipinski.NumRotatableBonds(mol)
                    hetero_atoms = Descriptors.NumHeteroatoms(mol)
                    hba = Lipinski.NumHAcceptors(mol)
                    hbd = Lipinski.NumHDonors(mol)
                    num_rings = rdMolDescriptors.CalcNumRings(mol)
                    return [molwt, tpsa, logp, rot_bonds, hetero_atoms, hba, hbd, num_rings]
                except Exception as e:
                    return [None]*8
            
            prop_names = [
                "Molecular Weight (g/mol)", "Topological polar surface area(TPSA) (Å²)", "LogP", 
                "Rotatable Bonds", "Heteroatoms", "H-Bond Acceptors", 
                "H-Bond Donors", "Number of Rings"
            ]
            # ... inside your `if mol:` block after description section
            ref_polymer = "PE"
            ref_smiles = smiles_df.loc[ref_polymer, "SMILE_repeating_unit"]
            ref_props = calculate_properties(ref_smiles)

            sel_polymer = polymer_name_input
            sel_smiles = smiles_df.loc[sel_polymer, "SMILE_repeating_unit"]
            sel_props = calculate_properties(sel_smiles)

            prop_names = [
                "Molecular Weight (g/mol)", "TPSA (Å²)", "LogP", 
                "Rotatable Bonds", "Heteroatoms", "H-Bond Acceptors", 
                "H-Bond Donors", "Number of Rings"
            ]

            comp_df = pd.DataFrame({ref_polymer: ref_props, sel_polymer: sel_props}, index=prop_names)

            st.markdown("### Molecular Descriptor Comparison (Repeat Unit)")
            st.markdown(comp_df.to_markdown())
            


            # --------------------------------------------------------
        else:
            st.error("Invalid SMILES string for this polymer.")
    else:
        st.warning("Polymer name not found in SMILES database. Please check spelling.")

#rdkit end

# --- Function to show comparison of user input vs dataset values ---

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

# --- FORM INPUT ---
if st.session_state.input_saved:
    st.success(f"✅ Inputs saved for **{st.session_state.polymer}**.")
    with st.expander("Show saved feature inputs"):
        st.dataframe(pd.Series(st.session_state.user_vals, name="Value"))
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Edit inputs"):
            st.session_state.input_saved = False
    with c2:
        if st.button("Reset inputs"):
            st.session_state.polymer = ""
            st.session_state.user_vals = {c: 0.0 for c in features.columns}
            st.session_state.input_saved = False
    if st.session_state.input_saved and st.session_state.polymer:
        show_comparison()
else:
    with st.form("user_entry_form", clear_on_submit=False):
        st.markdown("**Enter features for polymer:**")
              
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
            polymer = polymer_name_input.upper().strip()
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
                show_comparison()

# --- NEAREST POLYMERS ---
st.markdown("### 📊 Nearest equivalent polymers based on structural features input")

if not st.session_state.get("input_saved"):
    st.info("Please complete and save the user entry above to compute nearest polymers.")
else:
    polymer = st.session_state.polymer

    # Build user feature vector aligned to dataset order
    feature_order = list(features.columns)
    user_feat = pd.Series(st.session_state.user_vals, dtype=float).reindex(feature_order)

    # Compute user property index (PPI) and Z-score normalized PPI
    user_ppi = user_feat.dot(properties)
    user_ppi.name = f"{polymer}-user"
    user_ppi_z = (user_ppi - ppi_mean) / ppi_std

    # Prepare data for nearest neighbors search in PPI z-score space
    props_for_nn = list(ppi_z.columns)
    X = ppi_z[props_for_nn].astype(float).values
    y = user_ppi_z[props_for_nn].astype(float).values.reshape(1, -1)

    # NearestNeighbors model to find 3 closest polymers
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(y)  # distances and indices have shape (1, n_neighbors)

    # Flatten arrays for easier handling
    distances = distances.flatten()
    indices = indices.flatten()

    # Extract neighbor names and distances
    neighbor_names = np.array(ppi_z.index)[indices]
    neighbor_dist = distances

    # Number of neighbors found
    n_neighbors_found = len(neighbor_names)

    # Sort neighbors by ascending distance
    order = np.argsort(neighbor_dist)[:n_neighbors_found]

    names_ordered = neighbor_names[order]
    dists_ordered = neighbor_dist[order]

    # Compute relative distances (nearest at 0)
    min_dist = dists_ordered[0]
    if min_dist <= 0:
        min_dist = 1e-6  # avoid division by zero

    rel_dist = [round(max(0, (d - min_dist) / min_dist), 3) for d in dists_ordered]

    # Plot horizontal bar chart of nearest polymers
    st.subheader("1. Nearest equivalent polymers based on input features")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names_ordered[::-1], rel_dist[::-1])
    ax.set_xlabel("Relative distance in PPI space (min = 0)")
    ax.set_ylabel("Polymer")
    ax.set_title("Nearest polymers")
    for y_i, v in enumerate(rel_dist[::-1]):
        ax.text(v, y_i, f" {v:.2f}", va="center")
    fig.tight_layout()
    st.pyplot(fig)

    # --- Structural Features of Nearest Polymers ---
    st.subheader("2. Structural features of nearest equivalent polymers in feature space")

    # Avoid duplicating user's polymer if present in nearest neighbor list
    nn_list = [p for p in names_ordered if p != polymer]

    # Extract feature data for nearest neighbors and user input
    nearest_poly_features = features.loc[nn_list].T  # Features as rows, polymers as columns
    user_feat.name = f"{polymer}-USER"

    try:
        data_col = features.loc[polymer].rename(f"{polymer}-DATA")
    except KeyError:
        st.error(f"Polymer '{polymer}' not found in dataset index.")
        data_col = pd.Series(index=features.columns, dtype=float, name=f"{polymer}-DATA")

    # Combine user input, database data, and nearest neighbors for comparison
    compare_T = pd.concat(
        [user_feat.to_frame(), data_col.to_frame(), nearest_poly_features],
        axis=1
    )

    st.dataframe(compare_T.style.format("{:.2f}"))

    # --- Properties of Nearest Polymers from Reference Library ---
    st.subheader("3. Properties of nearest polymers from reference library")

    neighbor_library = library.loc[names_ordered]
    neighbor_library_T = neighbor_library.T
    st.dataframe(neighbor_library_T.style.format("{:.2f}"))

    st.markdown("Note: The nearest polymers are determined based on Euclidean distance in the standardized property index space.")


# ===== Category-based polymer selection =====
st.markdown("### Polymer property hierarchy by category")

# Define categories and their polymers (in desired order)
categories_ordered = [
    ("Commodity", ["PE", "PP", "PS", "PVC", "ABS", "PMMA", "ASA"]),
    ("Engineering", ["PA12", "PA6", "PA66", "PBT", "PET", "POM-H", "POM-C",
                     "POK", "COC", "PC", "PAR", "PPE", "PA46", "PPS", "PARA"]),
    ("High-performance", ["PPA", "PEI", "PES", "PSU", "PAI", "PI", "LCP", "PEEK", "PEK", "PBI"]),
]

# Filter only polymers available in the dataset
available = set(ppi.index)
categories = [(cat, [p for p in plist if p in available]) for cat, plist in categories_ordered]

# Initialize session state for selection controls
for cat_name, _ in categories:
    key = f"sel_{cat_name.lower().replace(' ', '_').replace('-', '_')}"
    if key not in st.session_state:
        st.session_state[key] = []

# Top-level control buttons for quick select/clear all
c1, c2 = st.columns(2)
if c1.button("Select ALL"):
    for cat_name, plist in categories:
        st.session_state[f"sel_{cat_name.lower().replace(' ', '_').replace('-', '_')}"] = plist[:]
if c2.button("Clear ALL"):
    for cat_name, _ in categories:
        st.session_state[f"sel_{cat_name.lower().replace(' ', '_').replace('-', '_')}"] = []

# Display categories in three columns with selection controls
cols = st.columns(3)
selected_polymers_all = []
for (cat_name, plist), col in zip(categories, cols):
    key = f"sel_{cat_name.lower().replace(' ', '_').replace('-', '_')}"
    with col:
        st.subheader(cat_name)
        btn_c1, btn_c2 = st.columns(2)
        if btn_c1.button("Select All", key=f"select_all_{key}"):
            st.session_state[key] = plist[:]
        if btn_c2.button("Clear", key=f"clear_{key}"):
            st.session_state[key] = []
        selected = st.multiselect(
            f"Select {cat_name.lower()} polymers",
            options=plist,
            default=st.session_state[key],
            key=key
        )
        selected_polymers_all.append(selected)

# Combine selected polymers across all categories, preserving order and uniqueness
from collections import OrderedDict
sel_polymers = list(OrderedDict.fromkeys(p for group in selected_polymers_all for p in group))

# Pick property subset (limit to 8 for readability)
st.markdown("### Pick indicative properties for visualization")
property_options = list(ppi.columns)
max_props = 8
if len(property_options) > max_props:
    property_options = property_options[:max_props]

if "sel_props" not in st.session_state:
    st.session_state.sel_props = property_options[:]

col_prop1, col_prop2 = st.columns(2)
with col_prop1:
    if st.button("Select All Properties"):
        st.session_state.sel_props = property_options[:]
with col_prop2:
    if st.button("Clear Properties"):
        st.session_state.sel_props = []

selected_props = st.multiselect(
    "Select properties",
    options=list(ppi.columns),
    default=st.session_state.sel_props,
    key="sel_props"
)

# Display Heatmap if selections are valid
if sel_polymers and selected_props:
    subset_ppi = ppi.loc[sel_polymers, selected_props]
    st.markdown("### Polymer Property Index Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(subset_ppi, annot=True, cmap="RdYlBu", center=0,
                annot_kws={"size": 8, "color": "grey"}, ax=ax)
    ax.set_xlabel("Property")
    ax.set_ylabel("Polymer")
    ax.set_title("Polymer Property Index — Selected Polymers & Properties")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
else:
    st.info("Please select at least one polymer and one property to display.")

# Random sample heatmap of 5 polymers & 3 properties
st.markdown("### Random polymer property rankings")
random_polymers = ppi.sample(n=5, random_state=42)
random_props = np.random.choice(ppi.columns, size=3, replace=False)
random_subset = random_polymers.loc[:, random_props]

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(random_subset, annot=True, cmap="RdYlBu", center=0,
            annot_kws={"size": 8, "color": "grey"}, ax=ax)
ax.set_title("Random Selection: Polymer Property Rankings")
ax.set_xlabel("Property")
ax.set_ylabel("Polymer")
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

# Polymer Feature-Property Correlation Matrix
st.markdown("### Polymer Feature-Property Correlation")
corr_matrix = pd.DataFrame({
    prop: [features[f].corr(ppi[prop]) for f in features.columns]
    for prop in ppi.columns
}, index=features.columns)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="RdYlBu", center=0,
            annot_kws={"size": 7, "color": "grey"}, ax=ax)
ax.set_title("Correlation between Structural Features and Polymer Properties")
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

# Sign map of positive, negative and neutral correlations
st.markdown("### Sign Map of structural features vs polymer properties")
sign_matrix = corr_matrix.applymap(lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))

color_map = {
    "Positive": "background-color: #d73027; color: white;",  # RdYlBu red/orange
    "Negative": "background-color: #4575b4; color: white;",  # RdYlBu blue
    "Neutral": "background-color: #ffffbf; color: black;",   # RdYlBu yellow/neutral
}


def style_sign_cells(val):
    return color_map.get(val, "")

styled_sign_matrix = sign_matrix.style.applymap(style_sign_cells)
st.dataframe(styled_sign_matrix)

# Top 3 features per property - positive and negative correlations
st.markdown("### Top 3 structural features influencing property")
top_features_list = []
for prop in corr_matrix.columns:
    series = corr_matrix[prop].dropna()
    top_pos = series[series > 0].nlargest(3)
    top_neg = series[series < 0].nsmallest(3)
    
    for feat, val in top_pos.items():
        top_features_list.append({"Property": prop, "Feature": feat, "Correlation": val, "Direction": "Positive"})
    for feat, val in top_neg.items():
        top_features_list.append({"Property": prop, "Feature": feat, "Correlation": val, "Direction": "Negative"})

top_features_df = pd.DataFrame(top_features_list)
st.dataframe(top_features_df)

# --- Cumulative Positive/Negative Contribution per Property ---

st.markdown("### 📊 Positive vs Negative contribution of structural features to property index")

# Split correlations into positive and negative contributions per property
pos_contrib = corr_matrix.clip(lower=0).sum(axis=0)
neg_contrib = corr_matrix.clip(upper=0).abs().sum(axis=0)

# Normalize to percentage contributions summing to 100%
total_contrib = pos_contrib + neg_contrib
pos_pct = (pos_contrib / total_contrib * 100).fillna(0)
neg_pct = (neg_contrib / total_contrib * 100).fillna(0)

# Limit to first 8 properties for clarity
pos_neg_pct_df = pd.DataFrame({
    "Positive": pos_pct,
    "Negative": neg_pct
}).loc[pos_pct.index[:8]]

# Plot stacked horizontal bar chart for overall positive/negative contribution per property
fig, ax = plt.subplots(figsize=(8, 5))
pos_neg_pct_df.plot(kind='barh', stacked=True, color=['red', 'yellow'], ax=ax)

ax.set_xlabel("Contribution (%)")
ax.set_ylabel("Properties")
ax.set_title("Relative Positive vs Negative Contribution of structural features to properties")
ax.legend(loc='lower right')

fig.tight_layout()
st.pyplot(fig)

# --- Detailed contribution for a fixed subset of features and properties ---

import math

# Select fixed set of 9 features and 8 properties
features_subset = features.columns[:9]
properties_subset = corr_matrix.columns[:8]

# Subset correlation matrix
corr_subset = corr_matrix.loc[features_subset, properties_subset]

# Calculate sum of positive and negative correlations per property
pos_sum = corr_subset.clip(lower=0).sum(axis=0)
neg_sum = corr_subset.clip(upper=0).abs().sum(axis=0)

# Prepare summary dataframe
summary_df = pd.DataFrame({
    "Positive": pos_sum,
    "Negative": neg_sum
}, index=properties_subset)


st.markdown("### Conclusion")

st.markdown(
    """
    <div style='text-align: justify;'>
    <p>
    This project attempts to offer a streamlined model that visually connects polymer repeat unit features to key material properties, 
    offering rapid comparison across commodity, engineering, and high-performance polymers. By quantifying and weighting 
    structural impacts, the approach enhances both predictive accuracy and interpretability complementing fundamental 
    group contribution methods. Notably, the model’s intuitive visualization serves as an educative 
    tool—empowering users to understand, compare, and predict polymer properties based on structural design,
    while deepening their grasp of structure‑property relationships.
    </p>
    <p>
    A practical next step would be to extend the model with curated experimental data and broader polymer types, which could 
    improve its accuracy and utility for property prediction and comparison. Such developments would support continual 
    learning and wider applicability in educational and research contexts. 
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<hr style='margin-top: 0; margin-bottom: 4px;'>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center; font-size: small; margin-top: 0; margin-bottom: 0;'>***End of Analysis***</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: right; font-size: small;'>PolyeXplore Polymer Property Visualization © polyeXplore — Sibabrata De</p>", unsafe_allow_html=True)

# EOF

