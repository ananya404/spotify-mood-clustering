import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Spotify Mood Clustering",
    page_icon="🎵",
    layout="wide"
)

# ── Load model, scaler, data ──────────────────────────────
@st.cache_resource
def load_assets():
    with open('kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    df = pd.read_csv('spotify_clustered.csv')
    return model, scaler, df

kmeans, scaler, df = load_assets()

features = ['danceability', 'energy', 'tempo', 'loudness', 'valence']

cluster_moods = {
    0: '🎉 Party / Happy',
    1: '😢 Sad / Acoustic',
    2: '🎸 Energetic / Hype',
    3: '😌 Chill / Neutral'
}

mood_colors = {
    '🎉 Party / Happy':    '#FFE66D',
    '😢 Sad / Acoustic':   '#FF6B6B',
    '🎸 Energetic / Hype': '#1DB954',
    '😌 Chill / Neutral':  '#4ECDC4'
}

# ── Header ────────────────────────────────────────────────
st.title('🎵 Spotify Music Mood Classifier')
st.markdown('##### Predict the mood of any song using its audio features')
st.divider()

# ── Layout: 2 columns ─────────────────────────────────────
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader('🎛️ Adjust Song Features')
    st.markdown('Move the sliders to match your song:')

    danceability = st.slider('💃 Danceability',  0.0, 1.0, 0.5, 0.01)
    energy       = st.slider('⚡ Energy',         0.0, 1.0, 0.5, 0.01)
    tempo        = st.slider('🥁 Tempo (BPM)',    50.0, 220.0, 120.0, 1.0)
    loudness     = st.slider('🔊 Loudness (dB)', -40.0, 0.0, -10.0, 0.1)
    valence      = st.slider('😊 Valence',        0.0, 1.0, 0.5, 0.01)

    st.divider()

    # Predict button
    if st.button('🎯 Predict Mood', use_container_width=True):
        input_data = np.array([[danceability, energy, tempo, loudness, valence]])
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]
        mood = cluster_moods[cluster]
        color = mood_colors[mood]

        st.session_state['predicted_mood'] = mood
        st.session_state['predicted_cluster'] = cluster
        st.session_state['input_scaled'] = input_scaled

# ── Result ────────────────────────────────────────────────
with col2:
    st.subheader('🎯 Prediction Result')

    if 'predicted_mood' in st.session_state:
        mood    = st.session_state['predicted_mood']
        cluster = st.session_state['predicted_cluster']
        color   = mood_colors[mood]

        # Mood badge
        st.markdown(
            f"""
            <div style='background-color:{color}; padding:20px; border-radius:15px; text-align:center;'>
                <h1 style='color:black; margin:0;'>{mood}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('')

        # Feature bar chart
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')

        vals = [danceability, energy, tempo/220, (loudness+40)/40, valence]
        labels = ['Danceability', 'Energy', 'Tempo', 'Loudness', 'Valence']
        ax.barh(labels, vals, color=color, alpha=0.85, edgecolor='none')

        ax.set_xlim(0, 1)
        ax.set_title('Your Song Profile', color='white', fontsize=11)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_visible(False)

        st.pyplot(fig)

        st.divider()

        # Similar songs
        st.subheader('🎵 Similar Songs in This Cluster')
        similar = (df[df['Cluster'] == cluster]
                   [['track_name', 'artist_name']]
                   .drop_duplicates()
                   .sample(min(8, len(df[df['Cluster'] == cluster])))
                   .reset_index(drop=True))
        similar.index += 1
        similar.columns = ['Track Name', 'Artist']
        st.dataframe(similar, use_container_width=True)

    else:
        st.info('👈 Adjust the sliders and click **Predict Mood** to get started!')

# ── PCA Plot ──────────────────────────────────────────────
st.divider()
st.subheader('🗺️ Where Does Your Song Fall? — PCA View')

pca = PCA(n_components=2, random_state=42)
features_data = df[features].values
X_pca = pca.fit_transform(scaler.transform(features_data))

fig2, ax2 = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor('#0E1117')
ax2.set_facecolor('#0E1117')

colors_list = ['#FFE66D', '#FF6B6B', '#1DB954', '#4ECDC4']
for i, (cluster_id, mood) in enumerate(cluster_moods.items()):
    mask = df['Cluster'] == cluster_id
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors_list[i], label=mood,
                alpha=0.3, s=8, edgecolors='none')

# Plot user's song if predicted
if 'input_scaled' in st.session_state:
    user_pca = pca.transform(st.session_state['input_scaled'])
    ax2.scatter(user_pca[0, 0], user_pca[0, 1],
                c='white', s=200, zorder=5,
                edgecolors='black', linewidths=2,
                marker='*', label='⭐ Your Song')

ax2.set_title('Song Clusters — PCA', color='white', fontsize=12)
ax2.tick_params(colors='white')
ax2.legend(fontsize=8, markerscale=2,
           facecolor='#1E1E1E', labelcolor='white')
for spine in ax2.spines.values():
    spine.set_visible(False)

st.pyplot(fig2)

# ── Footer ────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with KMeans Clustering on 232,725 Spotify songs</p>",
    unsafe_allow_html=True
)
