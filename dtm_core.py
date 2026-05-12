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
# CONFIGURATION — Edit everything here, the rest of the code stays unchanged
# =============================================================================
LEMMATIZE     = True             # True to enable lemmatization
SPACY_MODEL   = "en_core_web_sm" # spaCy model ("fr_core_news_sm" for French)

MODE       = "csv"          # "simulate" or "csv"

# --- Language (stopwords + preprocessing) ---
LANGUAGE = "english"   # "english" or "french"

# --- CSV parameters (ignored if MODE="simulate") ---
DATA_PATH     = "DataSet/data_nyt_5k_sample.csv"       # Path to your CSV file
COLUMN_DATE   = "date"           # Date column (format YYYY-MM-DD)
COLUMN_TEXT   = "content"        # Text column
GRANULARITY   = "Y"              # 'Y'=year, 'M'=month, 'Q'=quarter
MAX_FEATURES  = 100000              # Vocabulary size
MIN_DF        = 10                # Ignore words appearing in fewer than N docs
MAX_DF        = 1.0            # Ignore words appearing in more than X% of docs

# --- Model parameters ---
N_TOPICS    = 5
EPOCHS      = 1500
LR          = 0.02
SIGMA2      = 0.05              # Topic transition variance (beta)
DELTA2      = 0.05               # Proportion transition variance (alpha)

# --- Simulation-only parameters ---
T_SIM = 20
K_SIM = 5
V_SIM = 90
D_SIM = 100


# =============================================================================
# 1. DTM MODEL — identical to youpi.py (this is the core, do not modify)
# =============================================================================

class DynamicTopicModel(nn.Module):
    def __init__(self, num_topics, vocab_size, num_times, sigma2=0.01, delta2=0.05,
                 beta_init=None, alpha_init=None):
        """
        beta_init  : [K, V]  initial log-probabilities from LDA — optional
        alpha_init : [T, K]  initial temporal proportions — optional
        """
        super().__init__()
        self.K = num_topics
        self.V = vocab_size
        self.T = num_times

        self.sigma2 = torch.tensor(sigma2)
        self.delta2 = torch.tensor(delta2)

        # Variational parameters — beta (topics) and alpha (proportions)
        self.beta_hat      = nn.Parameter(torch.randn(self.K, self.T, self.V) * 0.01)
        self.log_beta_nu2  = nn.Parameter(torch.zeros(self.K, self.T))
        self.alpha_hat     = nn.Parameter(torch.randn(self.T, self.K) * 0.01)
        self.log_alpha_nu2 = nn.Parameter(torch.ones(self.T) * -2.0)

        # --- LDA initialization (breaks topic symmetry from the start) ---
        if beta_init is not None:
            # beta_init [K, V] → same value broadcast across all time slices
            with torch.no_grad():
                self.beta_hat.data = (
                    beta_init.unsqueeze(1).expand(-1, self.T, -1).clone()
                )
        if alpha_init is not None:
            with torch.no_grad():
                self.alpha_hat.data = alpha_init.clone()

        # Kalman filter initial conditions
        self.m0       = torch.zeros(self.V)
        self.V0       = torch.tensor(1.0)
        self.m0_alpha = torch.zeros(self.K)

    # -------------------------------------------------------------------------
    # Kalman filter: forward and backward (Appendix A, Blei 2006)
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
    # ELBO — variational lower bound (Eq. 4–5, Blei 2006)
    # corpus_counts : [T, K, V]  (counts per time, topic, word)
    # -------------------------------------------------------------------------
    def compute_elbo(self, corpus_counts):
        elbo = torch.tensor(0.0)
        beta_nu2 = torch.exp(self.log_beta_nu2).clamp(min=1e-12)

        # --- 1. Beta terms (topics) ---
        for k in range(self.K):
            m, V       = self.kalman_forward(self.beta_hat[k], beta_nu2[k], self.sigma2)
            m_t, V_t   = self.kalman_backward(m, V, self.sigma2)

            # Prior (smooth evolution of topics over time)
            diff  = m_t[1:] - m_t[:-1]
            prior = -0.5 * torch.sum(
                diff**2 + V_t[1:].unsqueeze(-1) + V_t[:-1].unsqueeze(-1)
            ) / self.sigma2
            entropy = 0.5 * torch.sum(torch.log(V_t + 1e-12))

            # Likelihood — Zeta bound (Appendix A)
            zeta    = torch.exp(m_t + 0.5 * V_t.unsqueeze(-1)).sum(dim=-1)
            n_tk_w  = corpus_counts[:, k, :]                        # [T, V]
            log_lik = torch.sum(n_tk_w * m_t) - torch.sum(n_tk_w.sum(-1) * torch.log(zeta))

            elbo = elbo + prior + entropy + log_lik

        # --- 2. Alpha terms (topic proportions) ---
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
    # Training loop
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

            # --- Robust stopping criterion (moving average + patience) ---
            EPS = 1e-3

            

                # --- Stopping criterion (standard norm) ---
            if len(history) > 1:
                # Relative change between current and previous epoch
                change = abs(history[-1] - history[-2]) / (abs(history[-2]) + 1e-12)
                
                # Patience can be lower since the measure is instantaneous
                EPS = 1e-5  
                
                if change < EPS:
                    patience_counter += 1
                else:
                    patience_counter = 0

                # On autorise l'arrêt dès que le critère est stable
                if patience_counter >= 5: 
                    print(f"  Convergence reached at epoch {epoch} (delta < {EPS}).")
                    break

    

        return history


# =============================================================================
# 2. SIMULATION — unchanged from youpi.py
# =============================================================================

def simulate_data(T=10, K=3, V=50, D_per_t=100):
    """Generates a synthetic corpus following the DTM generative process."""
    beta  = torch.zeros(K, T, V)
    alpha = torch.zeros(T, K)

    for t in range(1, T):
        beta[:, t, :]  = beta[:, t-1, :] + torch.randn(K, V) * 0.1
        alpha[t, :]    = alpha[t-1, :] + torch.randn(K) * 0.3

    # corpus_counts[t, k, v] = number of occurrences of word v in topic k at time t
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
# 3. CSV LOADING — minimal and robust
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
        # tagger required for POS (lemmatization), ner for entity blacklist
        nlp = spacy.load(spacy_model, disable=["parser"])
    except OSError:
        raise OSError(
            f"spaCy model '{spacy_model}' not installed. "
            f"Run: python -m spacy download {spacy_model}"
        )

    if progress_callback: progress_callback(5, "Loading CSV...")
    else: print("  Loading CSV...")
    df = pd.read_csv(path, parse_dates=[col_date])
    df = df.dropna(subset=[col_date, col_text])
    df[col_text] = df[col_text].astype(str)

    # -------------------------
    # Basic cleaning + lowercase
    # -------------------------
    def clean_text(text):
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"[^a-zA-ZàâçéèêëîïôûùüÿñæœÀÂÇÉÈÊËÎÏÔÛÙÜŸÑÆŒ\s]", " ", text)
        return text.strip()  # pas de lower() ici : le NER a besoin des majuscules

    df[col_text] = df[col_text].apply(clean_text)

    # -------------------------
    # Remove noisy n-grams (regex, robust to lowercasing)
    # 
    # context is kept, then filtered by CUSTOM_STOPWORDS if needed.
    # -------------------------
    NGRAM_BLACKLIST = [
        r"\bnew\s+york\s+times\b",
        r"\bnew\s+york\b",
        r"\bnytimes\b",
    ]

    def remove_blacklisted_ngrams(text):
        for pattern in NGRAM_BLACKLIST:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()

    if progress_callback: progress_callback(20, "Lemmatization in progress (spaCy)...")
    else: print(" Lemmatization in progress (spaCy)...")
    df[col_text] = df[col_text].apply(remove_blacklisted_ngrams)

    # -------------------------
    # Lemmatization
    # -------------------------
    if LEMMATIZE:
        if progress_callback: progress_callback(35, "Lemmatization in progress (spaCy)...")
        else: print("Lemmatization in progress (spaCy)...")
        texts = []
        for doc in nlp.pipe(df[col_text], batch_size=500):
            # Tokens appartenant à une entité NER → forme originale (pas de lemmatisation)
            ner_tokens = {token.i for ent in doc.ents for token in ent}
            tokens = []
            for token in doc:
                if not token.is_alpha or token.is_space:
                    continue
                if token.i in ner_tokens:
                    tokens.append(token.text.lower())   # nom propre : forme brute
                else:
                    tokens.append(token.lemma_.lower()) # mot commun : lemme
            texts.append(" ".join(tokens))

        df[col_text] = texts
        if progress_callback: progress_callback(55, "Lemmatization completed.")
        else: print("  Lemmatization completed.")
    else:
        # Pas de lemmatisation mais on lowercase quand même
        df[col_text] = df[col_text].str.lower()

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

    # Too generic / source-specific words: exclude from vocabulary
    CUSTOM_STOPWORDS = {
        "york", "new", "times", "nytimes",   # NYT residuals
        "say", "said", "mr", "ms", "mrs",    # journalistic verbs/titles
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

    if progress_callback: progress_callback(65, f"Vocabulary : {V} words — LDA Initialization...")
    else: print(f"  Vocabulary : {V} words")

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
    if progress_callback: progress_callback(90, "LDA completed — Corpus preprocessing in progress...")

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
# 4. KALMAN HELPERS — used by all plots
# =============================================================================

def get_smoothed_beta(model, k):
    """Smoothed means and variances for topic k → m_t [T,V], V_t [T]."""
    with torch.no_grad():
        m, V_f    = model.kalman_forward(
            model.beta_hat[k], torch.exp(model.log_beta_nu2[k]), model.sigma2
        )
        m_t, V_t  = model.kalman_backward(m, V_f, model.sigma2)
    return m_t, V_t          # [T, V], [T]


def get_smoothed_alpha(model):
    """Smoothed proportions (softmax) and variances → props [T,K], V_t [T]."""
    with torch.no_grad():
        m_a, V_a      = model.kalman_forward(
            model.alpha_hat, torch.exp(model.log_alpha_nu2), model.delta2, is_alpha=True
        )
        m_a_t, V_a_t  = model.kalman_backward(m_a, V_a, model.delta2)
        props          = torch.softmax(m_a_t, dim=-1).numpy()
    return props, V_a_t      # [T, K], [T]


# =============================================================================
# 5. VISUALIZATIONS — common (simulation + CSV)
# =============================================================================

# --- Plot A : ELBO ----------------------------------------------------------

def plot_elbo(history):
    """ELBO convergence curve."""
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
    """Horizontal bar chart of top words for each topic at a given time t."""
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

    label = "last period" if time_idx == -1 else f"period {time_idx}"
    plt.suptitle(f"Top {n_words} mots par topic ({label})", fontsize=14)
    plt.tight_layout()
    plt.show()


# --- Plot C : Évolution temporelle des topics (alpha) -----------------------

def plot_topic_evolution(model, time_labels=None):
    """Topic dominance curves over time."""
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
    """Stacked area chart of topic proportions over time."""
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
# 6. SIMULATION VISUALIZATIONS — true vs estimated
# =============================================================================

# --- Plot S1 : Word trajectories — estimated vs true (multi-topic) ----------

def plot_word_trajectories(model, true_beta, n_words=4):
    """
    For each topic: n_words words with their estimated log-prob trajectory
    (solid line + 1σ confidence interval) vs true (dashed).
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

        # Select words with the highest variance in true beta
        # (words that move the most are the most informative)
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


# --- Plot S2 : Alpha — estimated vs true --------------------------------------

def plot_alpha_true_vs_est(model, true_alpha):
    """
    Topic proportions over time: estimated vs true.
    One curve per topic, with the Kalman smoother confidence interval.
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


# --- Plot S3 : β heatmaps — estimated vs true side by side ---------------------

def plot_beta_heatmaps(model, true_beta, n_words=20):
    """
    For each topic: two heatmaps side by side.
    Left   = P(word | topic, t) estimated
    Right  = P(word | topic, t) true
    X axis = time, Y axis = top words
    """
    K      = model.K
    colors_list = ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds',
                   'YlOrBr', 'PuBu', 'BuGn', 'RdPu', 'GnBu']

    for k in range(K):
        m_t, _     = get_smoothed_beta(model, k)               # [T, V]
        probs_est  = torch.softmax(m_t, dim=-1).detach().numpy()   # [T, V]
        probs_true = torch.softmax(true_beta[k], dim=-1).numpy()   # [T, V]

        # Top words according to average true probabilities
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


# --- Plot S4 : Variational uncertainty ν² per topic ---------------------

def plot_kalman_uncertainty(model):
    """
    ν²_t (variational variance) of each topic over time.
    Indicates where the model is uncertain about its observations.
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
    ax.set_title("Variational uncertainty β (ν²_t)", fontweight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel("ν² (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Alpha (proportions)
    ax = axes[1]
    v_alpha = torch.exp(model.log_alpha_nu2).detach().numpy()  # [T]
    ax.plot(times, v_alpha, color='black', lw=2)
    ax.fill_between(times, 0, v_alpha, alpha=0.15, color='black')
    ax.set_yscale('log')
    ax.set_title("Variational uncertainty α (ν²_t)", fontweight='bold')
    ax.set_xlabel("Time")
    ax.set_ylabel("ν² (log scale)")
    ax.grid(alpha=0.3)

    plt.suptitle("Variational Kalman filter uncertainty", fontsize=13)
    plt.tight_layout()
    plt.show()


# --- Plot S5 : Residuals — estimated − true (alpha) ------------------------------

def plot_alpha_residuals(model, true_alpha):
    """
    Residuals per topic: (estimated proportion) − (true proportion).
    Helps identify if a topic is systematically over- or under-estimated.
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


# --- Plot S6 : Validation tiles — estimated vs true top words --------------

def plot_topic_tiles_validation(model, true_beta, vocab, n_words=6):
    """
    Grid (one panel per topic).
    Each panel: 2 side-by-side bar columns
      - Estimated (solid)  : top words according to the model
      - True      (hatched): top words according to true beta
    Allows checking whether dominant words are correctly recovered.
    """
    K      = model.K
    colors = cm.tab10.colors
    cols   = min(3, K)
    rows   = int(np.ceil(K / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for k in range(K):
        ax = axes[k // cols][k % cols]
        c  = colors[k % 10]

        # --- Estimated: probabilities at the last time period ---
        m_t, _     = get_smoothed_beta(model, k)
        probs_est  = torch.softmax(m_t[-1], dim=-1).detach().numpy()
        top_est    = np.argsort(probs_est)[-n_words:][::-1]

        # --- True: average probabilities over all time periods ---
        probs_true = torch.softmax(true_beta[k], dim=-1).mean(dim=0).numpy()
        top_true   = np.argsort(probs_true)[-n_words:][::-1]

        # Union of important words (estimated + true)
        all_idx    = list(dict.fromkeys(list(top_est) + list(top_true)))[:n_words * 2]
        labels     = [vocab[i] for i in all_idx]
        p_est      = probs_est[all_idx]
        p_true     = probs_true[all_idx]

        x    = np.arange(len(labels))
        w    = 0.38
        ax.barh(x + w/2, p_est,  w, color=c,     alpha=0.85, label="Estimated")
        ax.barh(x - w/2, p_true, w, color='grey', alpha=0.55,
                hatch='//', label="True")
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"Topic {k}", fontweight='bold')
        ax.set_xlabel("Probability")
        ax.legend(fontsize='x-small')
        ax.grid(axis='x', alpha=0.3)

    for k in range(K, rows * cols):
        axes[k // cols][k % cols].set_visible(False)

    plt.suptitle(
        "Top-word validation: Estimated (solid) vs True (hatched)",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()


# --- Plot S7 : Metrics table ----------------------------------------

def print_metrics(model, true_beta, true_alpha):
    """
    Prints a console table: MSE and Pearson correlation
    for beta and alpha, topic by topic.
    """
    from scipy.stats import pearsonr

    K = model.K
    print("\n" + "=" * 60)
    print(f"{'VALIDATION METRICS':^60}")
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
    print(f"{'Mean':<8} "
          f"{np.mean(mse_b_list):>10.5f} "
          f"{np.mean(corr_b_list):>10.3f} "
          f"{np.mean(mse_a_list):>10.5f} "
          f"{np.mean(corr_a_list):>10.3f}")
    print("=" * 60 + "\n")


# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # SIMULATION MODE
    # ------------------------------------------------------------------
    if MODE == "simulate":
        print("=== Simulation mode ===")
        corpus_counts, true_beta, true_alpha = simulate_data(
            T=T_SIM, K=K_SIM, V=V_SIM, D_per_t=D_SIM
        )
        K, T, V      = K_SIM, T_SIM, V_SIM
        vocab        = [f"word_{i}" for i in range(V)]
        time_labels  = list(range(T))
        beta_init    = None   # simulation: random initialization is sufficient
        alpha_init   = None   # (true assignments are known from corpus_counts)

    # ------------------------------------------------------------------
    # CSV MODE
    # ------------------------------------------------------------------
    elif MODE == "csv":
        print("=== Real data mode (CSV) ===")
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
        raise ValueError(f"MODE must be 'simulate' or 'csv', not '{MODE}'")

    print(f"\n  Corpus: T={T} slices | K={K} topics | V={V} words\n")

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    print(f"=== Training ({EPOCHS} epochs max) ===")
    model   = DynamicTopicModel(
        num_topics=K, vocab_size=V, num_times=T,
        sigma2=SIGMA2, delta2=DELTA2,
        beta_init=beta_init, alpha_init=alpha_init,
    )
    history = model.fit(corpus_counts, epochs=EPOCHS, lr=LR)

    # ------------------------------------------------------------------
    # COMMON VISUALIZATIONS
    # ------------------------------------------------------------------
    print("\n=== Common visualizations ===")
    plot_elbo(history)
    plot_top_words(model, vocab, n_words=8)
    plot_topic_evolution(model, time_labels)
    plot_stacked_topics(model, time_labels)

    # ------------------------------------------------------------------
    # SIMULATION VISUALIZATIONS — true vs estimated
    # ------------------------------------------------------------------
    if MODE == "simulate":
        print("\n=== Simulation diagnostics (true vs estimated) ===")

        # S1 — Word trajectories with CI
        plot_word_trajectories(model, true_beta, n_words=4)

        # S2 — Alpha proportions estimated vs true
        plot_alpha_true_vs_est(model, true_alpha)

        # S3 — Beta heatmaps side by side
        plot_beta_heatmaps(model, true_beta, n_words=20)

        # S4 — Variational uncertainty ν²
        plot_kalman_uncertainty(model)

        # S5 — Alpha residuals
        plot_alpha_residuals(model, true_alpha)

        # S6 — Top words estimated vs true tiles
        plot_topic_tiles_validation(model, true_beta, vocab, n_words=6)

        # S7 — Console metrics table
        print_metrics(model, true_beta, true_alpha)

# %%