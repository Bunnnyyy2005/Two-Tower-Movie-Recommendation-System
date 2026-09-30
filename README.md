# 🎬 Two-Tower Movie Recommendation System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Preprocessing-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)

</div>

A neural recommendation system that learns **which movies a user is
likely to enjoy** by learning a compact representation for both the user
and the movie.

Instead of treating recommendation as a simple rating-prediction
problem, this project frames it as a **Top-K retrieval problem**. The
model learns a user representation and a movie representation
separately, brings compatible pairs closer through a dot-product
interaction, and then ranks candidate movies for each user.

The project also brings in movie **genre information**, so the model is
not relying only on user--movie interaction history.

------------------------------------------------------------------------

## Why I built this

A recommendation system has a deceptively simple goal:

> Given what a user has interacted with before, which movies should
> appear at the top of their recommendation list?

I wanted to approach that problem with a model that is closer to the way
large-scale retrieval systems are designed, rather than building another
basic collaborative-filtering model.

The core idea is the **Two-Tower architecture**:

-   One tower learns a representation of the **user**.
-   The other learns a representation of the **movie**.
-   Their embeddings are compared using a dot product.
-   Movies can then be ranked according to how well they match the user.

------------------------------------------------------------------------

## 🧠 Project Workflow

The complete pipeline in the notebook follows this flow:

``` mermaid
flowchart LR
    A[Ratings Data] --> B[Data Cleaning & Validation]
    B --> C[Filter Users & Movies]
    C --> D[Create Positive Labels]
    D --> E[User & Movie Encoding]
    E --> F[Genre Feature Engineering]
    F --> G[Train / Validation / Test Split]
    G --> H[Negative Sampling]
    H --> I[Two-Tower Neural Network]
    I --> J[Model Training]
    J --> K[Top-K Candidate Ranking]
    K --> L[Precision@K / Recall@K / Hit Rate@K / NDCG@K]
    L --> M[Movie Recommendations]
```

### 1. Data preparation

The project starts with the rating interactions and movie metadata:

-   `ratings.csv`
-   `movies.csv`

The two datasets are merged using `movie_id`, giving the model access to
both interaction information and movie metadata.

The original dataset contains **105,339 user--movie interactions**.

To make the training data more reliable, users and movies with fewer
than **5 interactions** are filtered out.

After filtering:

-   **94,121 interactions**
-   **668 users**
-   **3,855 movies**
-   **51.4% positive interactions**

A rating of **4 or above** is treated as a positive interaction.

------------------------------------------------------------------------

### 2. User and movie encoding

Raw user IDs and movie IDs are converted into compact integer indices
using `LabelEncoder`.

This makes them suitable for embedding layers.

The result is a clean mapping:

``` text
User ID  →  User index  →  User embedding
Movie ID →  Movie index →  Movie embedding
```

------------------------------------------------------------------------

### 3. Movie genre features

Movie genres are converted into a **multi-hot representation**.

The dataset contains **19 unique genres**, producing a:

``` text
3,855 × 19
```

genre feature matrix.

This gives the item tower additional information beyond the movie ID
itself.

For example, a movie can be represented as:

``` text
Action     = 1
Comedy     = 0
Drama      = 1
Romance    = 0
Thriller   = 1
...
```

This helps the model learn from the content characteristics of movies
alongside interaction history.

------------------------------------------------------------------------

### 4. Train / validation / test preparation

The notebook creates:

-   **80,123 training interactions**
-   **4,583 validation interactions**
-   **9,415 test interactions**

Positive training interactions are then paired with sampled negative
movies.

The training generator uses a **4:1 negative-to-positive ratio**:

``` text
Positive samples : 34,427
Negative samples : 137,708
```

The negative samples represent movies that the user did not positively
interact with in the training data.

------------------------------------------------------------------------

# 🏗️ Two-Tower Model

The heart of the project is a neural Two-Tower recommender.

``` text
                         USER SIDE
                            │
                       User ID
                            │
                     Embedding(128)
                            │
                    Dense(128) + BN
                            │
                    Dense(64) + BN
                            │
                            ▼
                      User Vector
                            │
                            │
                         Dot Product
                            │
                            ▲
                      Item Vector
                            │
              ┌─────────────┴─────────────┐
              │                           │
        Movie ID Embedding          Genre Features
              │                           │
            128-D                    19-D multi-hot
              │                           │
              │                       Dense(64)
              └──────────┬────────────────┘
                         │
                    Concatenate
                         │
                  Dense(128) + BN
                         │
                   Dense(64) + BN
                         │
                         ▼
                     Item Vector
```

### User tower

The user tower learns a representation from:

-   User ID embedding: **128 dimensions**
-   Dense layer: **128 units**
-   Dense layer: **64 units**
-   Batch Normalization
-   Dropout

### Item tower

The movie tower combines:

-   Movie ID embedding: **128 dimensions**
-   Genre representation: **19 dimensions**
-   Genre projection: **64 dimensions**
-   Dense layer: **128 units**
-   Dense layer: **64 units**
-   Batch Normalization
-   Dropout

The final user and movie representations interact through a **dot
product**, followed by a sigmoid output.

### Model size

The final network contains:

**712,735 total parameters**

of which:

-   **638,722 trainable**
-   **74,013 non-trainable**

------------------------------------------------------------------------

# ⚙️ Training

The model is trained using:

-   **Optimizer:** Adam
-   **Learning rate:** `1e-4`
-   **Batch size:** `1024`
-   **Maximum epochs:** `150`
-   **Embedding dimension:** `128`
-   **Dropout:** `0.3`
-   **Negative sampling ratio:** `4`
-   **Random seed:** `42`

The loss function uses binary cross-entropy with **label smoothing** and
prediction clipping for numerical stability.

Training also uses:

-   Early stopping
-   Learning-rate reduction
-   Binary accuracy
-   Precision
-   Recall

------------------------------------------------------------------------

# 📊 Benchmark Results

The model is evaluated as a recommendation system rather than relying
only on classification accuracy.

For every test user, the model ranks the positive test movies together
with **500 sampled negative candidates** and evaluates the resulting
Top-K list.

The notebook evaluates **666 users**.

  Metric                   Top-5       Top-10       Top-20
  ----------------- ------------ ------------ ------------
  **Precision@K**     **0.2267**   **0.2068**   **0.1807**
  **Recall@K**        **0.1502**   **0.2492**   **0.3864**
  **Hit Rate@K**      **0.5976**   **0.7357**   **0.8378**
  **NDCG@K**          **0.2509**   **0.2729**   **0.3165**

### How to read these numbers

At **Top-20**, the system:

-   retrieves **38.64% of the relevant test items on average**
-   gives the user at least one relevant recommendation in **83.78% of
    evaluated cases**
-   achieves an **NDCG@20 of 0.3165**, reflecting the quality of the
    ranking positions

The increase in recall and hit rate as K grows is expected: the
recommender gets more opportunities to surface relevant movies when the
recommendation list becomes longer.

> **Evaluation note:** These results use 500 sampled negative candidates
> per user, not the complete movie catalog. They should therefore be
> interpreted as sampled-candidate ranking benchmarks.

------------------------------------------------------------------------

# 🛠️ Libraries & Tools

The notebook keeps the implementation relatively focused:

  Library                  Purpose
  ------------------------ -----------------------------------------------------
  **Python**               Core implementation
  **NumPy**                Numerical operations and feature matrices
  **Pandas**               Data loading, cleaning, filtering and preprocessing
  **TensorFlow / Keras**   Two-Tower neural network and model training
  **Scikit-learn**         User/movie label encoding
  **Matplotlib**           Training and metric visualisation

### Main techniques used

-   Embedding layers
-   Multi-hot genre encoding
-   Neural network towers
-   Batch Normalization
-   Dropout
-   Dot-product similarity
-   Negative sampling
-   Label smoothing
-   Adam optimization
-   Early stopping
-   Learning-rate scheduling
-   Top-K recommendation evaluation

------------------------------------------------------------------------

# 🔍 What makes the project interesting

The interesting part is not simply that a neural network predicts
whether a user will like a movie.

The project separates the problem into **user representation learning**
and **item representation learning**.

That design makes the system conceptually suitable for a retrieval
pipeline:

``` text
User
 ↓
User embedding
 ↓
Compare against movie representations
 ↓
Rank candidates
 ↓
Top-K recommendations
```

This is also why Two-Tower models are useful beyond movie
recommendation: the same pattern can be adapted to products, jobs,
advertisements, documents, videos, or other user--item retrieval
problems.

------------------------------------------------------------------------

# 📁 Project Structure

A simple version of the project can be organised as:

``` text
two-tower-recommender/
│
├── two-tower-method-final.ipynb
├── data/
│   ├── ratings.csv
│   └── movies.csv
│
├── models/
│   └── two_tower_model_2.h5
│
└── README.md
```

The notebook contains the complete experimentation workflow, from
preprocessing to recommendation generation.

------------------------------------------------------------------------

# 🚀 Recommended Upgrades

The current project is a solid working Two-Tower recommender, but there
are several changes that would make it much stronger as a serious Data
Science portfolio project.

### 1. Add strong baseline models

Before showing the neural model, compare it against simple and classical
approaches:

-   Most Popular recommender
-   User/item popularity baseline
-   Matrix Factorization
-   Neural Two-Tower model

This would answer an important Data Science question:

> **Is the additional complexity of the neural model actually buying us
> better recommendations?**

------------------------------------------------------------------------

### 2. Make the evaluation full-catalog

The current evaluation ranks each user's positives against **500 sampled
negative movies**.

A stronger experiment would rank against the entire eligible movie
catalog.

With only **3,855 movies**, this is completely practical.

Then the project can report:

``` text
Full-Catalog Precision@K
Full-Catalog Recall@K
Full-Catalog Hit Rate@K
Full-Catalog NDCG@K
```

That would make the reported numbers much easier to interpret and
compare.

------------------------------------------------------------------------

### 3. Add an ablation study for genres

Run two versions of the model:

``` text
Two-Tower
     vs
Two-Tower + Genre Features
```

Then measure how much the genre information actually contributes.

This turns:

> "I added genre features"

into:

> "I tested whether genre features improve recommendation quality."

That is a much stronger Data Science story.

------------------------------------------------------------------------

### 4. Add proper user-level error analysis

Break recommendation performance down by different user groups:

-   Light users vs heavy users
-   Popular movies vs less popular movies
-   Different genre preferences
-   Users with very few historical interactions
-   Users with strong or diverse preferences

This would help explain **where the model works and where it
struggles**, instead of stopping at a single overall metric.

------------------------------------------------------------------------

### 5. Add retrieval scalability with FAISS

Once the Two-Tower model produces item embeddings, store them in a
vector index such as **FAISS**.

The final pipeline would look more like:

``` text
User
 ↓
User Tower
 ↓
User Embedding
 ↓
FAISS Approximate Nearest Neighbour Search
 ↓
Top-K Movie Candidates
```

This would demonstrate the main practical advantage of a Two-Tower
architecture: user and item representations can be computed separately,
allowing fast candidate retrieval when the item catalog becomes very
large.

------------------------------------------------------------------------

# 🎯 Final Takeaway
 
This project started with a simple question --- **"What movies should I
recommend to this user?"** --- and turned it into a complete neural
recommendation pipeline.

The important pieces are all here:

**data preparation → feature engineering → negative sampling →
representation learning → candidate ranking → Top-K evaluation →
recommendations**

The current implementation already demonstrates a good understanding of
neural recommendation systems.

The next step is not necessarily to make the network deeper or more
complicated. The bigger improvement would come from making the
**experiments more rigorous**: stronger baselines, cleaner evaluation,
ablation studies, and deeper error analysis.

That would turn this from a good Two-Tower implementation into a much
more convincing **Data Science recommendation-system case study**.
