import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score

st.title("🌋 Geyser ML Clustering & Dimensionality Reduction App")

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = sns.load_dataset('geyser')
st.write("### Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# 2. Preprocessing
# -------------------------------
x = df.drop(columns=['kind'])
y = df['kind']
scaled = StandardScaler()
x_scaled = scaled.fit_transform(x)

# -------------------------------
# 3. PCA
# -------------------------------
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
pca_label = KMeans(n_clusters=2, random_state=42).fit_predict(x_pca)

# -------------------------------
# 4. LDA
# -------------------------------
lda = LinearDiscriminantAnalysis(n_components=1)
x_lda = lda.fit_transform(x_scaled, y)
lda_label = KMeans(n_clusters=2, random_state=42).fit_predict(x_lda)

# -------------------------------
# 5. t-SNE
# -------------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
x_tsne = tsne.fit_transform(x_scaled)
tsne_label = KMeans(n_clusters=2, random_state=42).fit_predict(x_tsne)

# -------------------------------
# 6. UMAP
# -------------------------------
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
x_umap = reducer.fit_transform(x_scaled)
umap_label = KMeans(n_clusters=2, random_state=42).fit_predict(x_umap)

# -------------------------------
# 7. Tabs for Visualization
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Original", "PCA", "LDA", "t-SNE", "UMAP"])

with tab1:
    st.write("### Original Scaled Data")
    fig, ax = plt.subplots()
    sns.scatterplot(x=x['duration'], y=x['waiting'], hue=y, ax=ax)
    st.pyplot(fig)

with tab2:
    st.write("### PCA Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1], hue=pca_label, palette="Set1", ax=ax)
    st.pyplot(fig)

with tab3:
    st.write("### LDA Projection")
    fig, ax = plt.subplots()
    sns.stripplot(x=x_lda[:,0], hue=lda_label, palette="Set1", jitter=0.2, ax=ax)
    st.pyplot(fig)

with tab4:
    st.write("### t-SNE Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_tsne[:,0], y=x_tsne[:,1], hue=tsne_label, palette="Set1", ax=ax)
    st.pyplot(fig)

with tab5:
    st.write("### UMAP Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_umap[:,0], y=x_umap[:,1], hue=umap_label, palette="Set1", ax=ax)
    st.pyplot(fig)

# -------------------------------
# 8. Silhouette Score Comparison
# -------------------------------
scores = {
    "PCA": silhouette_score(x_pca, pca_label),
    "LDA": silhouette_score(x_lda, lda_label),
    "t-SNE": silhouette_score(x_tsne, tsne_label),
    "UMAP": silhouette_score(x_umap, umap_label)
}
st.write("### 📊 Silhouette Scores")
st.dataframe(pd.DataFrame(scores.items(), columns=["Method", "Score"]))
# -------------------------------
# 9. Master Comparison Grid
# -------------------------------
st.write("## 🔍 Master Comparison Grid")

fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# 1. Original Data
sns.scatterplot(x=x['duration'], y=x['waiting'], hue=y, ax=axes[0,0])
axes[0,0].set_title("Original Data (True Label)")

sns.scatterplot(x=x['duration'], y=x['waiting'], hue=KMeans(n_clusters=2, random_state=42).fit_predict(x_scaled), palette="Set1", ax=axes[0,1])
axes[0,1].set_title("Original Data (KMeans Cluster)")

# 2. PCA
sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1], hue=y, ax=axes[1,0])
axes[1,0].set_title("PCA (True Label)")

sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1], hue=pca_label, palette="Set1", ax=axes[1,1])
axes[1,1].set_title("PCA (KMeans Cluster)")

# 3. LDA
sns.stripplot(x=x_lda[:,0], hue=y, jitter=0.2, alpha=0.8, size=6, ax=axes[2,0])
axes[2,0].set_title("LDA 1D (True Label)")

sns.stripplot(x=x_lda[:,0], hue=lda_label, palette="Set1", jitter=0.2, alpha=0.8, size=6, ax=axes[2,1])
axes[2,1].set_title("LDA 1D (KMeans Cluster)")

plt.tight_layout()
st.pyplot(fig)

# -------------------------------
# 10. Extra Grid for t-SNE & UMAP
# -------------------------------
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))

# t-SNE
sns.scatterplot(x=x_tsne[:,0], y=x_tsne[:,1], hue=y, ax=axes2[0,0])
axes2[0,0].set_title("t-SNE (True Label)")

sns.scatterplot(x=x_tsne[:,0], y=x_tsne[:,1], hue=tsne_label, palette="Set1", ax=axes2[0,1])
axes2[0,1].set_title("t-SNE (KMeans Cluster)")

# UMAP
sns.scatterplot(x=x_umap[:,0], y=x_umap[:,1], hue=y, ax=axes2[1,0])
axes2[1,0].set_title("UMAP (True Label)")

sns.scatterplot(x=x_umap[:,0], y=x_umap[:,1], hue=umap_label, palette="Set1", ax=axes2[1,1])
axes2[1,1].set_title("UMAP (KMeans Cluster)")

plt.tight_layout()
st.pyplot(fig2)
