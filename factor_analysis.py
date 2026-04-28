#s16798

"""
factor_analysis.py

Run this ONCE before opening the Streamlit factor analysis page.

    cd : s16798/python factor_analysis.py

Reads  : car_price_dataset .csv          (same folder — note the space)
         best_model.pkl                  (produced by train_model.py)
Saves  : factor_analysis_results.pkl    (loaded by the Streamlit factor page)

Install requirements (if needed):
    pip install prince scikit-learn matplotlib seaborn pandas numpy
"""

#Import libraries
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import prince

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

#File paths

#automatically get the folder where Python file is located.
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, "car_price_dataset .csv")  
OUTPUT_PATH  = os.path.join(BASE_DIR, "factor_analysis_results.pkl")

#Province mapping (mirrors train_model.py exactly) 
TOWN_TO_PROVINCE = {
    "Colombo":"Western","Gampaha":"Western","Negombo":"Western",
    "Kalutara":"Western","Panadura":"Western","Moratuwa":"Western",
    "Dehiwala-Mount-Lavinia":"Western","Maharagama":"Western",
    "Kotte":"Western","Wattala":"Western","Ja-Ela":"Western",
    "Kelaniya":"Western","Kadawatha":"Western","Nugegoda":"Western",
    "Piliyandala":"Western","Boralesgamuwa":"Western",
    "Kandy":"Central","Matale":"Central","Nuwara-Eliya":"Central",
    "Gampola":"Central","Nawalapitiya":"Central","Hatton":"Central",
    "Galle":"Southern","Matara":"Southern","Hambantota":"Southern",
    "Weligama":"Southern","Tangalle":"Southern","Hikkaduwa":"Southern",
    "Ambalangoda":"Southern","Jaffna":"Northern","Vavuniya":"Northern",
    "Kilinochchi":"Northern","Mullaitivu":"Northern",
    "Batticaloa":"Eastern","Trincomalee":"Eastern","Ampara":"Eastern",
    "Kalmunai":"Eastern","Kurunegala":"North Western","Puttalam":"North Western",
    "Kuliyapitiya":"North Western","Chilaw":"North Western",
    "Anuradapura":"North Central","Polonnaruwa":"North Central",
    "Badulla":"Uva","Bandarawela":"Uva","Haputale":"Uva","Welimada":"Uva",
    "Ratnapura":"Sabaragamuwa","Kegalle":"Sabaragamuwa","Balangoda":"Sabaragamuwa",
}

THRESHOLD = 20   # minimum listings — rarer categories grouped as "Other"


#Data loading (mirrors train_model.py exactly)

#Load and prepare dataset (load_and_prepare)
#path is the input argument to the function.
#str means string type.
#-> pd.DataFrame   -  This is called a return type hint.
#It tells readers that the function returns a DataFrame from pandas.

def load_and_prepare(path: str) -> pd.DataFrame:
    
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.dropna().drop_duplicates().copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Age"]  = 2026 - df["YOM"]

    for col in ["AIR CONDITION", "POWER STEERING", "POWER MIRROR", "POWER WINDOW"]:
        df[col + "_bin"] = (df[col] == "Available").astype(int)

    df["Leasing_bin"]   = (df["Leasing"] != "No Leasing").astype(int)
    df["Gear_bin"]      = (df["Gear"] == "Automatic").astype(int)
    df["Condition_bin"] = (df["Condition"] == "NEW").astype(int)

    for col in ["Brand", "Model", "Town"]:
        df[col] = df[col].astype(str).str.strip().str.title()
    df = df[(df["Brand"] != "") & (df["Model"] != "")]

    df["Province"] = df["Town"].map(TOWN_TO_PROVINCE).fillna("Other")

    for col in ["Brand", "Model", "Fuel Type"]:
        counts  = df[col].value_counts()
        rare    = counts[counts < THRESHOLD].index
        df[col] = df[col].replace(rare, "Other")

    return df


#Build FAMD-ready dataset
def build_famd_dataset(df: pd.DataFrame):
    """
    We include the same features used in the price model plus 'Price'.
    Binary encoded columns stay numerical (0/1).
    """
    numerical_cols = [
        "Engine (cc)", "Millage(KM)",
        "Gear_bin", "Leasing_bin",
        "AIR CONDITION_bin", "POWER STEERING_bin",
        "POWER MIRROR_bin", "POWER WINDOW_bin",
        "Condition_bin",
    ]
    categorical_cols = ["Fuel Type", "Province"]

    famd_df = df[numerical_cols + categorical_cols].copy()

    # Ensure correct dtypes
    for c in numerical_cols:
        famd_df[c] = pd.to_numeric(famd_df[c], errors="coerce")
    for c in categorical_cols:
        famd_df[c] = famd_df[c].astype(str)

    famd_df = famd_df.dropna().reset_index(drop=True)
    return famd_df, numerical_cols, categorical_cols


#save figure to bytes (for pickle storage)
def fig_to_bytes(fig) -> bytes:
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return data


#Scree + Cumulative Variance
def plot_scree(eigenvalues, explained_variance_ratio, cumulative_explained_variance):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scree plot
    ax = axes[0]
    ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker="o", color="steelblue")
    ax.axhline(y=1, color="red", linestyle="--", label="Kaiser Criterion (λ = 1)")
    ax.set_title("Scree Plot (FAMD)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Eigenvalue")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    # Cumulative variance
    ax2 = axes[1]
    ax2.plot(range(1, len(cumulative_explained_variance) + 1),
             cumulative_explained_variance, marker="o", color="darkorange")
    ax2.axhline(y=0.90, color="green",  linestyle="--", label="90% Variance")
    ax2.axhline(y=0.95, color="orange", linestyle="--", label="95% Variance")
    ax2.set_title("Cumulative Explained Variance (FAMD)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    return fig


#Score Plot (Factor 1 vs Factor 2)
def plot_score(famd_scores, explained_variance_ratio):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        famd_scores.iloc[:, 0], famd_scores.iloc[:, 1],
        alpha=0.4, color="steelblue", s=20, edgecolors="none"
    )
    ax.set_title("FAMD Score Plot (Factor 1 vs Factor 2)", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"Factor 1 ({explained_variance_ratio[0]*100:.2f}%)")
    ax.set_ylabel(f"Factor 2 ({explained_variance_ratio[1]*100:.2f}%)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    return fig


# Silhouette Score vs k
def plot_silhouette_line(famd_components, K_range):
    silhouette_scores = []
    for k in K_range:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(famd_components)
        silhouette_scores.append(silhouette_score(famd_components, labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(K_range, silhouette_scores, marker="o", linewidth=2, color="steelblue")
    ax.scatter(K_range, silhouette_scores, color="red", zorder=5, s=60)

    best_k   = K_range[int(np.argmax(silhouette_scores))]
    best_sil = max(silhouette_scores)
    ax.annotate(
        f"Best k={best_k}\n({best_sil:.3f})",
        xy=(best_k, best_sil),
        xytext=(best_k + 0.4, best_sil - 0.02),
        arrowprops=dict(arrowstyle="->", color="darkred"),
        color="darkred", fontsize=10,
    )
    ax.set_title("Silhouette Score vs Number of Clusters", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(list(K_range))
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    return fig, silhouette_scores


# Combined Cluster Scatter + Silhouette plots for each k
def plot_clusters_and_silhouettes(famd_components, K_range):
    """Returns a list of (k, scatter_fig_bytes, silhouette_fig_bytes, sil_score)."""
    results = []
    comp1 = famd_components.iloc[:, 0] if hasattr(famd_components, "iloc") else famd_components[:, 0]
    comp2 = famd_components.iloc[:, 1] if hasattr(famd_components, "iloc") else famd_components[:, 1]

    for k in K_range:
        kmeans        = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(famd_components)
        sil_score     = silhouette_score(famd_components, cluster_labels)
        sil_samples   = silhouette_samples(famd_components, cluster_labels)

        # ── Scatter plot ──
        fig_scatter, ax_s = plt.subplots(figsize=(7, 5))
        scatter = ax_s.scatter(comp1, comp2, c=cluster_labels,
                               cmap="Set2", s=20, alpha=0.6, edgecolors="none")
        ax_s.set_title(f"K-Means Clusters (k={k})", fontsize=12, fontweight="bold")
        ax_s.set_xlabel("FAMD Component 1")
        ax_s.set_ylabel("FAMD Component 2")
        plt.colorbar(scatter, ax=ax_s, label="Cluster")
        ax_s.grid(True, linestyle=":", alpha=0.5)
        fig_scatter.tight_layout()
        scatter_bytes = fig_to_bytes(fig_scatter)

        #Silhouette plot ──
        fig_sil, ax_sil = plt.subplots(figsize=(7, 5))
        y_lower = 10
        cmap    = cm.get_cmap("Set2")
        for j in range(k):
            vals  = sil_samples[cluster_labels == j]
            vals.sort()
            y_upper = y_lower + len(vals)
            color   = cmap(j / k)
            ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                                 facecolor=color, edgecolor=color, alpha=0.7)
            ax_sil.text(-0.05, y_lower + 0.5 * len(vals), str(j), fontsize=9)
            y_lower = y_upper + 10

        ax_sil.axvline(x=sil_score, color="red", linestyle="--",
                       label=f"Avg score = {sil_score:.3f}")
        ax_sil.set_title(f"Silhouette Plot (k={k}, score={sil_score:.3f})",
                         fontsize=12, fontweight="bold")
        ax_sil.set_xlabel("Silhouette Coefficient")
        ax_sil.set_ylabel("Cluster")
        ax_sil.set_xlim([-0.1, 1])
        ax_sil.set_ylim([0, len(famd_components) + (k + 1) * 10])
        ax_sil.legend(fontsize=9)
        ax_sil.grid(True, linestyle=":", alpha=0.5)
        fig_sil.tight_layout()
        sil_bytes = fig_to_bytes(fig_sil)

        results.append({
            "k":               k,
            "cluster_labels":  cluster_labels,
            "sil_score":       sil_score,
            "scatter_png":     scatter_bytes,
            "silhouette_png":  sil_bytes,
        })

    return results


# Main fuction 
def run():
    print(f"Loading data: {DATA_PATH}")
    df = load_and_prepare(DATA_PATH)
    print(f"Rows after cleaning: {len(df)}")

    famd_train, numerical_cols, categorical_cols = build_famd_dataset(df)
    print(f"FAMD dataset shape   : {famd_train.shape}")
    print(f"Numerical features   : {numerical_cols}")
    print(f"Categorical features : {categorical_cols}")

    # Fit FAMD
    print("\nFitting FAMD …")
    famd = prince.FAMD(
        n_components=famd_train.shape[1],
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=42,
        engine="sklearn",
        handle_unknown="error",
    )
    famd = famd.fit(famd_train)

    eigenvalues                  = famd.eigenvalues_
    explained_variance_ratio     = eigenvalues / np.sum(eigenvalues)
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    n_kaiser = int(np.sum(eigenvalues > 1))
    n_90     = int(np.argmax(cumulative_explained_variance >= 0.90) + 1)
    n_95     = int(np.argmax(cumulative_explained_variance >= 0.95) + 1)

    print("===================================")
    print(f"Original features      : {famd_train.shape[1]}")
    print(f"Kaiser criterion (λ>1) : {n_kaiser} components")
    print(f"90% explained variance : {n_90} components")
    print(f"95% explained variance : {n_95} components")
    print("===================================")

    # Scree + Cumulative 
    print("Generating scree plot …")
    fig_scree = plot_scree(eigenvalues, explained_variance_ratio, cumulative_explained_variance)
    scree_png = fig_to_bytes(fig_scree)

    #Factor scores
    famd_scores    = famd.transform(famd_train)         # n × n_components
    famd_components = famd.row_coordinates(famd_train)  # alias used for clustering

    #Score plot 
    print("Generating score plot …")
    fig_score = plot_score(famd_scores, explained_variance_ratio)
    score_png = fig_to_bytes(fig_score)

    #Silhouette line
    K_range = range(2, 11)
    print("Computing silhouette scores for k=2…10 …")
    fig_sil_line, silhouette_scores_list = plot_silhouette_line(famd_components, K_range)
    sil_line_png = fig_to_bytes(fig_sil_line)

    best_k = list(K_range)[int(np.argmax(silhouette_scores_list))]
    print(f"Best k by silhouette   : {best_k}")

    # Cluster + Silhouette for each k 
    print("Generating per-k cluster & silhouette plots …")
    cluster_results = plot_clusters_and_silhouettes(famd_components, K_range)

    #Factor loadings (column coordinates)
    try:
        column_coords = famd.column_coordinates(famd_train)
    except Exception:
        column_coords = None

    #Save everything 
    payload = {
        # Raw numbers
        "eigenvalues":                   eigenvalues,
        "explained_variance_ratio":      explained_variance_ratio,
        "cumulative_explained_variance":  cumulative_explained_variance,
        "n_kaiser": n_kaiser,
        "n_90":     n_90,
        "n_95":     n_95,
        "n_features": famd_train.shape[1],
        "famd_scores":      famd_scores,
        "famd_components":  famd_components,
        "column_coordinates": column_coords,
        "silhouette_scores":  silhouette_scores_list,
        "best_k":             best_k,

        # Pre-rendered PNGs (bytes)
        "scree_png":     scree_png,
        "score_png":     score_png,
        "sil_line_png":  sil_line_png,

        # Per-k cluster results (each has scatter_png, silhouette_png, etc.)
        "cluster_results": cluster_results,

        # Metadata
        "numerical_cols":    numerical_cols,
        "categorical_cols":  categorical_cols,
        "n_rows":            len(famd_train),
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nfactor_analysis_results.pkl saved - {OUTPUT_PATH}")
    print("   You can now open the Factor Analysis page in the Streamlit app.")


if __name__ == "__main__":
    run()