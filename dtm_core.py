# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy 

torch.set_default_dtype(torch.float64)
np.random.seed(42)

# =============================================================================
# CONFIGURATION — Tout modifier ici, le reste du code ne change pas
# =============================================================================
LEMMATIZE     = True             # True pour activer la lemmatisation
SPACY_MODEL   = "en_core_web_sm" # Modèle spaCy ("fr_core_news_sm" pour le français)

MODE       = "csv"          # "simulate" ou "csv"

# --- Langue (stopwords + preprocessing) ---
LANGUAGE = "english"   # "english" ou "french"

# --- Paramètres CSV (ignorés si MODE="simulate") ---
DATA_PATH     = "DataSet/data_nyt_5k_sample.csv"       # Chemin vers votre fichier CSV
COLUMN_DATE   = "date"           # Colonne date (format YYYY-MM-DD)
COLUMN_TEXT   = "content"        # Colonne texte
GRANULARITY   = "Y"              # 'Y'=année, 'M'=mois, 'Q'=trimestre
MAX_FEATURES  = 100000              # Taille du vocabulaire
MIN_DF        = 10                # Ignorer les mots dans moins de N docs
MAX_DF        = 0.2             # Ignorer les mots dans plus de X% des docs

# --- Paramètres du modèle ---
N_TOPICS    = 5
EPOCHS      = 1500
LR          = 0.02
SIGMA2      = 0.05              # Variance de transition des topics (beta)
DELTA2      = 0.05               # Variance de transition des proportions (alpha)

# --- Paramètres simulation uniquement ---
T_SIM = 20
K_SIM = 5
V_SIM = 90
D_SIM = 100


# =============================================================================
# 1. MODÈLE DTM — identique à youpi.py (c'est le cœur, on ne touche pas)
# =============================================================================

class DynamicTopicModel(nn.Module):
    def __init__(self, num_topics, vocab_size, num_times, sigma2=0.01, delta2=0.05,
                 beta_init=None, alpha_init=None):
        """
        beta_init  : [K, V]  log-probabilités initiales issues de LDA — optionnel
        alpha_init : [T, K]  proportions temporelles initiales — optionnel
        """
        super().__init__()
        self.K = num_topics
        self.V = vocab_size
        self.T = num_times

        self.sigma2 = torch.tensor(sigma2)
        self.delta2 = torch.tensor(delta2)

        # Paramètres variationnels — beta (topics) et alpha (proportions)
        self.beta_hat      = nn.Parameter(torch.randn(self.K, self.T, self.V) * 0.01)
        self.log_beta_nu2  = nn.Parameter(torch.zeros(self.K, self.T))
        self.alpha_hat     = nn.Parameter(torch.randn(self.T, self.K) * 0.01)
        self.log_alpha_nu2 = nn.Parameter(torch.ones(self.T) * -2.0)

        # --- Initialisation LDA (brise la symétrie entre topics dès le départ) ---
        if beta_init is not None:
            # beta_init [K, V] → même valeur sur toutes les tranches temporelles
            with torch.no_grad():
                self.beta_hat.data = (
                    beta_init.unsqueeze(1).expand(-1, self.T, -1).clone()
                )
        if alpha_init is not None:
            with torch.no_grad():
                self.alpha_hat.data = alpha_init.clone()

        # Conditions initiales du filtre de Kalman
        self.m0       = torch.zeros(self.V)
        self.V0       = torch.tensor(1.0)
        self.m0_alpha = torch.zeros(self.K)

    # -------------------------------------------------------------------------
    # Filtre de Kalman : forward et backward (Appendix A, Blei 2006)
    # -------------------------------------------------------------------------
    def kalman_forward(self, obs_hat, obs_nu2, transition_sigma2, is_alpha=False):
        m_list, V_list = [], []
        m_prev = self.m0_alpha if is_alpha else self.m0
        V_prev = self.V0

        for t in range(self.T):
            denom = V_prev + transition_sigma2 + obs_nu2[t]
            V_t   = (obs_nu2[t] / denom) * (V_prev + transition_sigma2)
            m_t   = (obs_nu2[t] / denom) * m_prev + (1 - obs_nu2[t] / denom) * obs_hat[t]
            m_list.append(m_t)
            V_list.append(V_t)
            m_prev, V_prev = m_t, V_t

        return torch.stack(m_list), torch.stack(V_list)

    def kalman_backward(self, m, V, transition_sigma2):
        m_tilde_list = [m[-1]]
        V_tilde_list = [V[-1]]

        for t in range(self.T - 2, -1, -1):
            ratio     = V[t] / (V[t] + transition_sigma2)
            m_t_tilde = ratio * m[t] + (1 - ratio) * m_tilde_list[0]
            V_t_tilde = V[t] + (ratio**2) * (V_tilde_list[0] - (V[t] + transition_sigma2))
            m_tilde_list.insert(0, m_t_tilde)
            V_tilde_list.insert(0, V_t_tilde)

        return torch.stack(m_tilde_list), torch.stack(V_tilde_list)

    # -------------------------------------------------------------------------
    # ELBO — borne variationnelle (Eq. 4–5, Blei 2006)
    # corpus_counts : [T, K, V]  (counts par temps, topic, mot)
    # -------------------------------------------------------------------------
    def compute_elbo(self, corpus_counts):
        elbo = torch.tensor(0.0)
        beta_nu2 = torch.exp(self.log_beta_nu2).clamp(min=1e-12)

        # --- 1. Termes Beta (topics) ---
        for k in range(self.K):
            m, V       = self.kalman_forward(self.beta_hat[k], beta_nu2[k], self.sigma2)
            m_t, V_t   = self.kalman_backward(m, V, self.sigma2)

            # Prior (évolution lisse des topics dans le temps)
            diff  = m_t[1:] - m_t[:-1]
            prior = -0.5 * torch.sum(
                diff**2 + V_t[1:].unsqueeze(-1) + V_t[:-1].unsqueeze(-1)
            ) / self.sigma2
            entropy = 0.5 * torch.sum(torch.log(V_t + 1e-12))

            # Vraisemblance — borne Zeta (Appendix A)
            zeta    = torch.exp(m_t + 0.5 * V_t.unsqueeze(-1)).sum(dim=-1)
            n_tk_w  = corpus_counts[:, k, :]                        # [T, V]
            log_lik = torch.sum(n_tk_w * m_t) - torch.sum(n_tk_w.sum(-1) * torch.log(zeta))

            elbo = elbo + prior + entropy + log_lik

        # --- 2. Termes Alpha (proportions de topics) ---
        alpha_nu2     = torch.exp(self.log_alpha_nu2).clamp(min=1e-12)
        m_a, V_a      = self.kalman_forward(self.alpha_hat, alpha_nu2, self.delta2, is_alpha=True)
        m_a_t, V_a_t  = self.kalman_backward(m_a, V_a, self.delta2)

        alpha_prior = -0.5 * torch.sum(
            (m_a_t[1:] - m_a_t[:-1])**2
            + V_a_t[1:].unsqueeze(-1)
            + V_a_t[:-1].unsqueeze(-1)
        ) / self.delta2
        alpha_entropy = 0.5 * torch.sum(torch.log(V_a_t + 1e-12))

        zeta_alpha   = torch.exp(m_a_t + 0.5 * V_a_t.unsqueeze(-1)).sum(dim=-1)  # [T]
        n_tk_total   = corpus_counts.sum(dim=-1)                                   # [T, K]
        alpha_log_lik = (
            torch.sum(n_tk_total * m_a_t)
            - torch.sum(n_tk_total.sum(-1) * torch.log(zeta_alpha))
        )

        return elbo + alpha_prior + alpha_entropy + alpha_log_lik

    # -------------------------------------------------------------------------
    # Boucle d'entraînement
    # -------------------------------------------------------------------------
    def fit(self, corpus_counts, epochs=1500, lr=1e-2):
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        history   = []
        patience_counter = 0

        for epoch in range(epochs):
            optimizer.zero_grad()
            elbo = self.compute_elbo(corpus_counts)
            (-elbo).backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            optimizer.step()

            history.append(elbo.item())

            # --- Critère d'arrêt robuste (moving average + patience) ---
            EPS = 1e-3

            

                # --- Critère d'arrêt (Norme classique) ---
            if len(history) > 1:
                # Changement relatif entre l'époque actuelle et la précédente
                change = abs(history[-1] - history[-2]) / (abs(history[-2]) + 1e-12)
                
                # On peut baisser la patience car la mesure est instantanée
                EPS = 1e-5  
                
                if change < EPS:
                    patience_counter += 1
                else:
                    patience_counter = 0

                # On autorise l'arrêt dès que le critère est stable
                if patience_counter >= 5: 
                    print(f"  Convergence atteinte à l'époque {epoch} (delta < {EPS}).")
                    break

    

        return history


# =============================================================================
# 2. SIMULATION — inchangée par rapport à youpi.py
# =============================================================================

def simulate_data(T=10, K=3, V=50, D_per_t=100):
    """Génère un corpus synthétique selon le processus génératif du DTM."""
    beta  = torch.zeros(K, T, V)
    alpha = torch.zeros(T, K)

    for t in range(1, T):
        beta[:, t, :]  = beta[:, t-1, :] + torch.randn(K, V) * 0.1
        alpha[t, :]    = alpha[t-1, :] + torch.randn(K) * 0.3

    # corpus_counts[t, k, v] = nb occurrences du mot v dans le topic k au temps t
    corpus_counts = torch.zeros(T, K, V)
    for t in range(T):
        theta = torch.softmax(alpha[t], dim=-1)
        for d in range(D_per_t):
            for k in range(K):
                n_words = int(60 * theta[k].item())
                if n_words > 0:
                    prob   = torch.softmax(beta[k, t], dim=-1)
                    counts = torch.multinomial(prob, n_words, replacement=True)
                    for w in counts:
                        corpus_counts[t, k, w] += 1

    return corpus_counts, beta, alpha


# =============================================================================
# 3. CHARGEMENT CSV — minimal et robuste
# =============================================================================

import spacy

def get_spacy_model(language: str):
    if language == "french":
        return "fr_core_news_sm"
    return "en_core_web_sm"


def load_csv_data(path, col_date, col_text, granularity, n_topics,
                  max_features, min_df, max_df, language="english",
                  progress_callback=None, date_start=None, date_end=None):
    """
    Clean + vectorize CSV text data for DTM.
    Returns:
      - corpus_counts [T, K, V]
      - vocab [V]
      - time_labels [T]
      - beta_init [K, V]
      - alpha_init [T, K]
    """

    # -------------------------
    # Language / spaCy model
    # -------------------------
    spacy_model = get_spacy_model(language)

    try:
        # tagger requis pour POS (lemmatisation), ner pour blacklist entités
        nlp = spacy.load(spacy_model, disable=["parser"])
    except OSError:
        raise OSError(
            f"spaCy model '{spacy_model}' not installed. "
            f"Run: python -m spacy download {spacy_model}"
        )

    if progress_callback: progress_callback(5, "Chargement du CSV...")
    else: print("  Chargement du CSV...")
    df = pd.read_csv(path, parse_dates=[col_date])
    df = df.dropna(subset=[col_date, col_text])
    df[col_text] = df[col_text].astype(str)

    # -------------------------
    # Nettoyage de base + lowercase
    # -------------------------
    def clean_text(text):
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"[^a-zA-Zàâçéèêëîïôûùüÿñæœ\s]", " ", text)
        return text.lower().strip()

    df[col_text] = df[col_text].apply(clean_text)

    # -------------------------
    # Suppression des n-grammes parasites (regex, robuste au lowercasing)
    # "new york" supprimé comme bigramme ; "new" seul dans un autre
    # contexte est conservé puis filtré par CUSTOM_STOPWORDS si besoin.
    # -------------------------
    NGRAM_BLACKLIST = [
        r"\bnew\s+york\s+times\b",
        r"\bnew\s+york\b",
        r"\bnytimes\b",
    ]

    def remove_blacklisted_ngrams(text):
        for pattern in NGRAM_BLACKLIST:
            text = re.sub(pattern, " ", text)
        return re.sub(r"\s+", " ", text).strip()

    if progress_callback: progress_callback(20, "Suppression des n-grammes parasites...")
    else: print("  Suppression des n-grammes parasites...")
    df[col_text] = df[col_text].apply(remove_blacklisted_ngrams)

    # -------------------------
    # Lemmatisation
    # -------------------------
    if LEMMATIZE:
        if progress_callback: progress_callback(35, "Lemmatisation en cours (spaCy)...")
        else: print("  Lemmatisation en cours (spaCy)...")
        texts = []
        for doc in nlp.pipe(df[col_text], batch_size=500):
            tokens = [
                token.lemma_.lower()
                for token in doc
                if token.is_alpha and not token.is_space
            ]
            texts.append(" ".join(tokens))

        df[col_text] = texts
        if progress_callback: progress_callback(55, "Lemmatisation terminée.")
        else: print("  Lemmatisation terminée.")

    df = df[df[col_text].str.strip() != ""]
    df = df.sort_values(col_date).reset_index(drop=True)

    # -------------------------
    # Time slicing
    # -------------------------
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df = df.dropna(subset=[col_date])

    df["time_slice"] = df[col_date].dt.to_period(granularity).astype(str)

    # ---- Temporal window filter ----
    if date_start is not None:
        df = df[df["time_slice"] >= date_start]
    if date_end is not None:
        df = df[df["time_slice"] <= date_end]
    df = df.reset_index(drop=True)

    time_labels = sorted(df["time_slice"].unique())
    time_map = {t: i for i, t in enumerate(time_labels)}
    df["t"] = df["time_slice"].map(time_map)

    T = len(time_labels)

    # -------------------------
    # Stopwords
    # -------------------------
    if language == "french":
        from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOP
        stop_words = set(FR_STOP)
    else:
        stop_words = set(ENGLISH_STOP_WORDS)

    # Mots trop génériques / liés à la source : à exclure du vocabulaire
    CUSTOM_STOPWORDS = {
        "york", "new", "times", "nytimes",   # résidus NYT
        "say", "said", "mr", "ms", "mrs",    # verbes/titres journalistiques
    }
    stop_words = stop_words | CUSTOM_STOPWORDS

    # -------------------------
    # Bag-of-words (IMPORTANT: counts, not TF-IDF)
    # -------------------------
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        stop_words=list(stop_words),
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )

    dtm = vectorizer.fit_transform(df[col_text]).toarray().astype(np.float64)
    vocab = vectorizer.get_feature_names_out()
    V = len(vocab)

    if progress_callback: progress_callback(65, f"Vocabulaire : {V} mots — Initialisation LDA...")
    else: print(f"  Vocabulaire : {V} mots")

    # -------------------------
    # LDA init (for DTM initialization)
    # -------------------------
    from sklearn.decomposition import LatentDirichletAllocation

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method="batch",
        random_state=42,
    )

    doc_topic = lda.fit_transform(dtm)
    if progress_callback: progress_callback(90, "LDA terminé — Préparation du corpus...")

    corpus_counts = torch.zeros(T, n_topics, V, dtype=torch.float64)

    t_idx_arr = df["t"].values

    for t_idx in range(T):
        mask = t_idx_arr == t_idx
        if mask.sum() == 0:
            continue
        theta_t = doc_topic[mask]
        counts_t = dtm[mask]
        corpus_counts[t_idx] = torch.tensor(theta_t.T @ counts_t)

    # -------------------------
    # init beta / alpha
    # -------------------------
    components_norm = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
    beta_init = torch.tensor(np.log(components_norm + 1e-10), dtype=torch.float64)

    alpha_init = torch.zeros((T, n_topics), dtype=torch.float64)

    time_labels = [str(t) for t in time_labels]

    return corpus_counts, vocab, time_labels, beta_init, alpha_init


# =============================================================================
# 4. HELPERS KALMAN — utilisés par tous les plots
# =============================================================================

def get_smoothed_beta(model, k):
    """Moyennes et variances lissées du topic k → m_t [T,V], V_t [T]."""
    with torch.no_grad():
        m, V_f    = model.kalman_forward(
            model.beta_hat[k], torch.exp(model.log_beta_nu2[k]), model.sigma2
        )
        m_t, V_t  = model.kalman_backward(m, V_f, model.sigma2)
    return m_t, V_t          # [T, V], [T]


def get_smoothed_alpha(model):
    """Proportions lissées (softmax) et variances → props [T,K], V_t [T]."""
    with torch.no_grad():
        m_a, V_a      = model.kalman_forward(
            model.alpha_hat, torch.exp(model.log_alpha_nu2), model.delta2, is_alpha=True
        )
        m_a_t, V_a_t  = model.kalman_backward(m_a, V_a, model.delta2)
        props          = torch.softmax(m_a_t, dim=-1).numpy()
    return props, V_a_t      # [T, K], [T]


# =============================================================================
# 5. VISUALISATIONS — communes (simulation + CSV)
# =============================================================================

# --- Plot A : ELBO ----------------------------------------------------------

def plot_elbo(history):
    """Courbe de convergence de l'ELBO."""
    plt.figure(figsize=(9, 3))
    plt.plot(history, color='steelblue', lw=1.5)
    plt.title("Convergence de l'ELBO", fontsize=13)
    plt.xlabel("Epochs")
    plt.ylabel("ELBO")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- Plot B : Top mots par topic --------------------------------------------

def plot_top_words(model, vocab, n_words=8, time_idx=-1):
    """Barres horizontales des top mots pour chaque topic à un instant t."""
    K     = model.K
    cols  = min(3, K)
    rows  = int(np.ceil(K / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    colors    = cm.tab10.colors

    for k in range(K):
        ax        = axes[k // cols][k % cols]
        m_t, _    = get_smoothed_beta(model, k)
        probs     = torch.softmax(m_t[time_idx], dim=-1).numpy()
        top_idx   = np.argsort(probs)[-n_words:][::-1]
        top_w     = [vocab[i] for i in top_idx]
        top_p     = probs[top_idx]

        ax.barh(range(n_words), top_p[::-1], color=colors[k % 10], alpha=0.8)
        ax.set_yticks(range(n_words))
        ax.set_yticklabels(top_w[::-1], fontsize=10)
        ax.set_title(f"Topic {k}", fontweight='bold')
        ax.set_xlabel("Probabilité")
        ax.grid(axis='x', alpha=0.3)

    for k in range(K, rows * cols):
        axes[k // cols][k % cols].set_visible(False)

    label = "dernière période" if time_idx == -1 else f"période {time_idx}"
    plt.suptitle(f"Top {n_words} mots par topic ({label})", fontsize=14)
    plt.tight_layout()
    plt.show()


# --- Plot C : Évolution temporelle des topics (alpha) -----------------------

def plot_topic_evolution(model, time_labels=None):
    """Courbes de dominance des topics au fil du temps."""
    props, _  = get_smoothed_alpha(model)
    T, K      = props.shape
    times     = [str(p) for p in time_labels] if time_labels is not None else list(range(T))
    colors    = cm.tab10.colors

    plt.figure(figsize=(12, 4))
    for k in range(K):
        plt.plot(times, props[:, k], label=f"Topic {k}", lw=2.5, color=colors[k % 10])

    plt.title("Évolution temporelle des topics (α lissé)", fontsize=13)
    plt.xlabel("Période")
    plt.ylabel("Proportion estimée")
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- Plot D : Distribution empilée ------------------------------------------

def plot_stacked_topics(model, time_labels=None):
    """Aire empilée des proportions de topics dans le temps."""
    props, _  = get_smoothed_alpha(model)
    T, K      = props.shape
    times     = [str(p) for p in time_labels] if time_labels is not None else list(range(T))
    colors    = cm.tab10.colors

    plt.figure(figsize=(12, 4))
    bottom = np.zeros(T)
    for k in range(K):
        plt.fill_between(range(T), bottom, bottom + props[:, k],
                         alpha=0.75, label=f"Topic {k}", color=colors[k % 10])
        bottom += props[:, k]

    plt.xticks(range(T), times, rotation=45, ha='right')
    plt.title("Distribution empilée des topics par période", fontsize=13)
    plt.ylabel("Proportion")
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 6. VISUALISATIONS SIMULATION — vrai vs estimé
# =============================================================================

# --- Plot S1 : Trajectoires de mots — estimé vs vrai (multi-topic) ----------

def plot_word_trajectories(model, true_beta, n_words=4):
    """
    Pour chaque topic : n_words mots avec leur trajectoire log-prob estimée
    (trait plein + intervalle de confiance à 1σ) vs vraie (pointillés).
    """
    K      = model.K
    T      = model.T
    times  = np.arange(T)
    colors = cm.tab10.colors

    cols = min(3, K)
    rows = int(np.ceil(K / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for k in range(K):
        ax         = axes[k // cols][k % cols]
        m_t, V_t   = get_smoothed_beta(model, k)               # [T,V], [T]
        std_t      = torch.sqrt(V_t).numpy()                    # [T]

        # Sélectionner les mots ayant la variance la plus forte dans le vrai beta
        # (les mots qui bougent le plus sont les plus informatifs)
        true_var   = true_beta[k].var(dim=0).numpy()            # [V]
        word_idx   = np.argsort(true_var)[-n_words:][::-1]

        for i, w in enumerate(word_idx):
            c      = colors[i % 10]
            m_np   = m_t[:, w].detach().numpy()
            true_np = true_beta[k, :, w].numpy()

            ax.plot(times, m_np, color=c, lw=2, label=f"mot {w} (est.)")
            ax.fill_between(times, m_np - std_t, m_np + std_t,
                            color=c, alpha=0.12)
            ax.plot(times, true_np, '--', color=c, lw=1.2, alpha=0.6,
                    label=f"mot {w} (vrai)")

        ax.set_title(f"Topic {k} — log-prob mots", fontweight='bold')
        ax.set_xlabel("Temps")
        ax.set_ylabel("log-prob (β)")
        ax.legend(fontsize='x-small', ncol=2)
        ax.grid(alpha=0.25)

    for k in range(K, rows * cols):
        axes[k // cols][k % cols].set_visible(False)

    plt.suptitle("Trajectoires β : Estimé (plein ± 1σ) vs Vrai (pointillés)", fontsize=14)
    plt.tight_layout()
    plt.show()


# --- Plot S2 : Alpha — estimé vs vrai --------------------------------------

def plot_alpha_true_vs_est(model, true_alpha):
    """
    Proportions de topics au fil du temps : estimé vs vrai.
    Une courbe par topic, avec l'intervalle de confiance du lisseur Kalman.
    """
    T      = model.T
    K      = model.K
    times  = np.arange(T)
    colors = cm.tab10.colors

    props_est, V_a_t   = get_smoothed_alpha(model)              # [T,K], [T]
    std_a              = torch.sqrt(V_a_t).numpy()              # [T]
    true_props         = torch.softmax(true_alpha, dim=-1).numpy()  # [T, K]

    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4), squeeze=False)
    for k in range(K):
        ax = axes[0][k]
        c  = colors[k % 10]

        ax.plot(times, props_est[:, k], color=c, lw=2.5, label="Estimé")
        ax.fill_between(times,
                        np.clip(props_est[:, k] - std_a * 0.1, 0, 1),
                        np.clip(props_est[:, k] + std_a * 0.1, 0, 1),
                        color=c, alpha=0.15)
        ax.plot(times, true_props[:, k], 'k--', lw=1.5, alpha=0.7, label="Vrai")

        ax.set_title(f"Topic {k}", fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_xlabel("Temps")
        if k == 0:
            ax.set_ylabel("Proportion (softmax α)")
        ax.legend(fontsize='small')
        ax.grid(alpha=0.3)

    plt.suptitle("Proportions α : Estimé vs Vrai par topic", fontsize=14)
    plt.tight_layout()
    plt.show()


# --- Plot S3 : Heatmaps β — estimé vs vrai côte à côte ---------------------

def plot_beta_heatmaps(model, true_beta, n_words=20):
    """
    Pour chaque topic : deux heatmaps côte à côte.
    Gauche  = P(mot | topic, t) estimé
    Droite  = P(mot | topic, t) vrai
    Axe X = temps, Axe Y = top mots
    """
    K      = model.K
    colors_list = ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds',
                   'YlOrBr', 'PuBu', 'BuGn', 'RdPu', 'GnBu']

    for k in range(K):
        m_t, _     = get_smoothed_beta(model, k)               # [T, V]
        probs_est  = torch.softmax(m_t, dim=-1).detach().numpy()   # [T, V]
        probs_true = torch.softmax(true_beta[k], dim=-1).numpy()   # [T, V]

        # Top mots selon les vraies probabilités moyennes
        top_idx    = np.argsort(probs_true.mean(axis=0))[-n_words:][::-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        cmap            = colors_list[k % len(colors_list)]

        im1 = ax1.imshow(probs_est[:, top_idx].T, aspect='auto', cmap=cmap,
                         interpolation='nearest')
        ax1.set_title(f"Topic {k} — Estimé", fontweight='bold')
        ax1.set_xlabel("Temps")
        ax1.set_ylabel("Mot (top)")
        ax1.set_yticks(range(n_words))
        ax1.set_yticklabels([f"mot_{i}" for i in top_idx], fontsize=8)
        plt.colorbar(im1, ax=ax1, label="P(mot|topic,t)")

        im2 = ax2.imshow(probs_true[:, top_idx].T, aspect='auto', cmap=cmap,
                         interpolation='nearest')
        ax2.set_title(f"Topic {k} — Vrai", fontweight='bold')
        ax2.set_xlabel("Temps")
        plt.colorbar(im2, ax=ax2, label="P(mot|topic,t)")

        plt.suptitle(f"Heatmap β — Topic {k} : Estimé vs Vrai", fontsize=13)
        plt.tight_layout()
        plt.show()


# --- Plot S4 : Incertitude variationnelle ν² par topic ---------------------

def plot_kalman_uncertainty(model):
    """
    ν²_t (variance variationnelle) de chaque topic au fil du temps.
    Indique où le modèle est incertain sur ses observations.
    """
    T      = model.T
    K      = model.K
    times  = np.arange(T)
    colors = cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Beta (topics)
    ax = axes[0]
    v_beta = torch.exp(model.log_beta_nu2).detach().numpy()   # [K, T]
    for k in range(K):
        ax.plot(times, v_beta[k], label=f"Topic {k}", color=colors[k % 10], lw=2)
    ax.set_yscale('log')
    ax.set_title("Incertitude variationnelle β (ν²_t)", fontweight='bold')
    ax.set_xlabel("Temps")
    ax.set_ylabel("ν² (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Alpha (proportions)
    ax = axes[1]
    v_alpha = torch.exp(model.log_alpha_nu2).detach().numpy()  # [T]
    ax.plot(times, v_alpha, color='black', lw=2)
    ax.fill_between(times, 0, v_alpha, alpha=0.15, color='black')
    ax.set_yscale('log')
    ax.set_title("Incertitude variationnelle α (ν²_t)", fontweight='bold')
    ax.set_xlabel("Temps")
    ax.set_ylabel("ν² (log scale)")
    ax.grid(alpha=0.3)

    plt.suptitle("Incertitude du filtre de Kalman variationnel", fontsize=13)
    plt.tight_layout()
    plt.show()


# --- Plot S5 : Résidus — estimé − vrai (alpha) ------------------------------

def plot_alpha_residuals(model, true_alpha):
    """
    Résidus par topic : (proportion estimée) − (proportion vraie).
    Aide à voir si un topic est systématiquement sur- ou sous-estimé.
    """
    T      = model.T
    K      = model.K
    times  = np.arange(T)
    colors = cm.tab10.colors

    props_est, _  = get_smoothed_alpha(model)
    true_props    = torch.softmax(true_alpha, dim=-1).numpy()

    plt.figure(figsize=(12, 4))
    for k in range(K):
        residuals = props_est[:, k] - true_props[:, k]
        plt.plot(times, residuals, label=f"Topic {k}",
                 color=colors[k % 10], lw=2)

    plt.axhline(0, color='black', lw=1, ls='--')
    plt.fill_between(times, -0.05, 0.05, color='grey', alpha=0.08,
                     label="±5% zone")
    plt.title("Résidus α : Estimé − Vrai par topic", fontsize=13)
    plt.xlabel("Temps")
    plt.ylabel("Résidu de proportion")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- Plot S6 : Tuiles de validation — top mots estimé vs vrai --------------

def plot_topic_tiles_validation(model, true_beta, vocab, n_words=6):
    """
    Grille (un panneau par topic).
    Dans chaque panneau : 2 colonnes de barres côte à côte
      - Estimé (plein)   : top mots selon le modèle
      - Vrai  (hachuré)  : top mots selon le vrai beta
    Permet de voir si les mots dominants sont bien récupérés.
    """
    K      = model.K
    colors = cm.tab10.colors
    cols   = min(3, K)
    rows   = int(np.ceil(K / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for k in range(K):
        ax = axes[k // cols][k % cols]
        c  = colors[k % 10]

        # --- Estimé : probabilités à la dernière période ---
        m_t, _     = get_smoothed_beta(model, k)
        probs_est  = torch.softmax(m_t[-1], dim=-1).detach().numpy()
        top_est    = np.argsort(probs_est)[-n_words:][::-1]

        # --- Vrai : probabilités moyennes sur toutes les périodes ---
        probs_true = torch.softmax(true_beta[k], dim=-1).mean(dim=0).numpy()
        top_true   = np.argsort(probs_true)[-n_words:][::-1]

        # Union des mots importants (estimés + vrais)
        all_idx    = list(dict.fromkeys(list(top_est) + list(top_true)))[:n_words * 2]
        labels     = [vocab[i] for i in all_idx]
        p_est      = probs_est[all_idx]
        p_true     = probs_true[all_idx]

        x    = np.arange(len(labels))
        w    = 0.38
        ax.barh(x + w/2, p_est,  w, color=c,     alpha=0.85, label="Estimé")
        ax.barh(x - w/2, p_true, w, color='grey', alpha=0.55,
                hatch='//', label="Vrai")
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"Topic {k}", fontweight='bold')
        ax.set_xlabel("Probabilité")
        ax.legend(fontsize='x-small')
        ax.grid(axis='x', alpha=0.3)

    for k in range(K, rows * cols):
        axes[k // cols][k % cols].set_visible(False)

    plt.suptitle(
        "Validation des top-mots : Estimé (plein) vs Vrai (hachuré)",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()


# --- Plot S7 : Tableau des métriques ----------------------------------------

def print_metrics(model, true_beta, true_alpha):
    """
    Affiche un tableau console : MSE et corrélation de Pearson
    pour beta et alpha, topic par topic.
    """
    from scipy.stats import pearsonr

    K = model.K
    print("\n" + "=" * 60)
    print(f"{'MÉTRIQUES DE VALIDATION':^60}")
    print("=" * 60)
    print(f"{'Topic':<8} {'MSE β':>10} {'Corr β':>10} {'MSE α':>10} {'Corr α':>10}")
    print("-" * 60)

    props_est, _  = get_smoothed_alpha(model)
    true_props    = torch.softmax(true_alpha, dim=-1).numpy()

    mse_a_list, corr_a_list = [], []
    mse_b_list, corr_b_list = [], []

    for k in range(K):
        m_t, _       = get_smoothed_beta(model, k)
        p_est        = torch.softmax(m_t, dim=-1).detach().numpy()     # [T, V]
        p_true       = torch.softmax(true_beta[k], dim=-1).numpy()     # [T, V]

        mse_b  = float(np.mean((p_est - p_true)**2))
        try:
            corr_b, _ = pearsonr(p_est.flatten(), p_true.flatten())
        except Exception:
            corr_b = float('nan')

        mse_a  = float(np.mean((props_est[:, k] - true_props[:, k])**2))
        try:
            corr_a, _ = pearsonr(props_est[:, k], true_props[:, k])
        except Exception:
            corr_a = float('nan')

        mse_b_list.append(mse_b)
        corr_b_list.append(corr_b)
        mse_a_list.append(mse_a)
        corr_a_list.append(corr_a)

        print(f"{k:<8} {mse_b:>10.5f} {corr_b:>10.3f} {mse_a:>10.5f} {corr_a:>10.3f}")

    print("-" * 60)
    print(f"{'Moyenne':<8} "
          f"{np.mean(mse_b_list):>10.5f} "
          f"{np.mean(corr_b_list):>10.3f} "
          f"{np.mean(mse_a_list):>10.5f} "
          f"{np.mean(corr_a_list):>10.3f}")
    print("=" * 60 + "\n")


# =============================================================================
# 7. EXÉCUTION PRINCIPALE
# =============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # MODE SIMULATION
    # ------------------------------------------------------------------
    if MODE == "simulate":
        print("=== Mode simulation ===")
        corpus_counts, true_beta, true_alpha = simulate_data(
            T=T_SIM, K=K_SIM, V=V_SIM, D_per_t=D_SIM
        )
        K, T, V      = K_SIM, T_SIM, V_SIM
        vocab        = [f"mot_{i}" for i in range(V)]
        time_labels  = list(range(T))
        beta_init    = None   # simulation : initialisation aléatoire suffisante
        alpha_init   = None   # (les vraies assignations sont connues dans corpus_counts)

    # ------------------------------------------------------------------
    # MODE CSV
    # ------------------------------------------------------------------
    elif MODE == "csv":
        print("=== Mode données réelles (CSV) ===")
        corpus_counts, vocab, time_labels, beta_init, alpha_init = load_csv_data(
            path        = DATA_PATH,
            col_date    = COLUMN_DATE,
            col_text    = COLUMN_TEXT,
            granularity = GRANULARITY,
            n_topics    = N_TOPICS,
            max_features= MAX_FEATURES,
            min_df      = MIN_DF,
            max_df      = MAX_DF,
        )
        K            = N_TOPICS
        T, _, V      = corpus_counts.shape
        true_beta    = None
        true_alpha   = None

    else:
        raise ValueError(f"MODE doit être 'simulate' ou 'csv', pas '{MODE}'")

    print(f"\n  Corpus : T={T} tranches | K={K} topics | V={V} mots\n")

    # ------------------------------------------------------------------
    # ENTRAÎNEMENT
    # ------------------------------------------------------------------
    print(f"=== Entraînement ({EPOCHS} epochs max) ===")
    model   = DynamicTopicModel(
        num_topics=K, vocab_size=V, num_times=T,
        sigma2=SIGMA2, delta2=DELTA2,
        beta_init=beta_init, alpha_init=alpha_init,
    )
    history = model.fit(corpus_counts, epochs=EPOCHS, lr=LR)

    # ------------------------------------------------------------------
    # VISUALISATIONS COMMUNES
    # ------------------------------------------------------------------
    print("\n=== Visualisations communes ===")
    plot_elbo(history)
    plot_top_words(model, vocab, n_words=8)
    plot_topic_evolution(model, time_labels)
    plot_stacked_topics(model, time_labels)

    # ------------------------------------------------------------------
    # VISUALISATIONS SIMULATION — vrai vs estimé
    # ------------------------------------------------------------------
    if MODE == "simulate":
        print("\n=== Diagnostics simulation (vrai vs estimé) ===")

        # S1 — Trajectoires de mots avec IC
        plot_word_trajectories(model, true_beta, n_words=4)

        # S2 — Proportions alpha estimé vs vrai
        plot_alpha_true_vs_est(model, true_alpha)

        # S3 — Heatmaps beta côte à côte
        plot_beta_heatmaps(model, true_beta, n_words=20)

        # S4 — Incertitude variationnelle ν²
        plot_kalman_uncertainty(model)

        # S5 — Résidus alpha
        plot_alpha_residuals(model, true_alpha)

        # S6 — Tuiles top mots estimé vs vrai
        plot_topic_tiles_validation(model, true_beta, vocab, n_words=6)

        # S7 — Tableau de métriques console
        print_metrics(model, true_beta, true_alpha)

# %%