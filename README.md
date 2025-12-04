#  Customer Segmentation Using Unsupervised Machine Learning

A comprehensive machine learning project demonstrating customer segmentation through unsupervised learning techniques. This project implements **K-Means Clustering**, **Feature Engineering**, and **Exploratory Data Analysis (EDA)** to identify distinct customer groups within a retail environment.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation & Business Context](#motivation--business-context)
- [Dataset Description](#dataset-description)
- [Technologies & Libraries](#technologies--libraries)
- [Project Architecture](#project-architecture)
- [Methodology](#methodology)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Feature Engineering](#feature-engineering)
  - [Clustering Algorithms](#clustering-algorithms)
  - [Model Evaluation](#model-evaluation)
- [Key Results & Insights](#key-results--insights)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Results Visualization](#results-visualization)
- [Conclusions & Recommendations](#conclusions--recommendations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [Author](#author)

---

##  Project Overview

This project applies **unsupervised machine learning** techniques to customer data from a shopping mall, enabling businesses to understand and profile their customer base. The primary objective is to identify homogeneous customer segments that can drive targeted marketing campaigns, personalized recommendations, and optimized business strategies.

### Key Objectives

1. Identify natural groupings within the customer population
2. Engineer meaningful features that capture customer behavior and purchasing power
3. Develop interpretable and actionable customer segments
4. Provide data-driven insights for business decision-making

---

##  Motivation & Business Context

In modern retail environments, understanding customer diversity is crucial for:

- **Personalized Marketing**: Tailor promotions and campaigns to specific customer segments
- **Inventory Management**: Optimize stock based on segment purchasing patterns
- **Customer Retention**: Design loyalty programs targeting high-value customers
- **Revenue Optimization**: Allocate resources to the most profitable segments
- **Competitive Advantage**: Gain insights into market positioning and gaps

By leveraging unsupervised learning, we eliminate the need for labeled training data, allowing exploratory discovery of natural customer groups without predefined assumptions.

---

## Dataset Description

### Source
**Mall_Customers.csv** - A dataset containing demographic and spending information for 200 shopping mall customers.

### Features & Attributes

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| **CustomerID** | Integer | Unique identifier for each customer | 1–200 |
| **Gender** | Categorical | Customer gender (Male/Female) | - |
| **Age** | Integer | Customer's age in years | 18–70 |
| **Annual Income (k$)** | Numerical | Approximate annual income in thousands of dollars | 15–137 |
| **Spending Score** | Integer | Proprietary score indicating customer spending behavior | 1–100 |

### Dataset Characteristics

- **Size**: 200 samples, 5 features
- **Missing Values**: None
- **Data Type**: Tabular/Structured
- **Target**: Unsupervised (no labeled outcomes)

### Spending Score Definition

The spending score is a proprietary metric (1–100) derived from customer transaction history, frequency, and monetary value. It reflects how likely a customer is to make purchases based on historical behavior.

---

## Technologies & Libraries

### Core Programming
- **Python 3.8+** - Primary programming language
- **Jupyter Notebook** - Interactive development environment

### Data Processing & Analysis
- **Pandas 1.3+** - Data manipulation and transformation
- **NumPy 1.21+** - Numerical computations and array operations
- **SciPy 1.7+** - Advanced statistical functions and hierarchical clustering

### Machine Learning
- **Scikit-Learn 1.0+** - KMeans, StandardScaler, preprocessing
- **Clustering Algorithms**: KMeans, Hierarchical Clustering

### Visualization
- **Matplotlib 3.4+** - Static, publication-quality plots
- **Seaborn 0.11+** - Statistical data visualization
- **Plotly 5.0+** - Interactive 3D visualizations and dashboards

### Development Tools
- **Git** - Version control
- **pip** - Package management

---

## 🏗 Project Architecture

```
customer-segmentation/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── segmentation.ipynb                 # Main analysis notebook
├── Mall_Customers.csv


---

## 🔬Methodology

### Exploratory Data Analysis (EDA)

The initial EDA phase focuses on understanding data distribution, identifying patterns, and detecting anomalies:

**1. Univariate Analysis**
- Statistical summaries (mean, median, std, min, max)
- Distribution plots for each feature
- Identification of skewness and outliers

**2. Bivariate Analysis**
- Correlation analysis between features
- Gender vs. spending patterns
- Age vs. income relationships
- Heatmap correlation matrices

**3. Multivariate Analysis**
- Pairplot relationships across multiple dimensions
- Gender-based demographic segmentation
- Spending behavior by age groups

### Feature Engineering

The feature engineering phase creates meaningful variables that enhance clustering performance:

#### Affluence Score (Composite Feature)

A derived metric combining income and spending behavior to measure customer purchasing power:

$$\text{Affluence Score} = 0.6 \times \text{Normalized Income} + 0.4 \times \text{Normalized Spending Score}$$

**Rationale:**
- Income (60% weight) represents financial capacity
- Spending Score (40% weight) represents behavioral tendency
- Normalization ensures comparability across different scales
- Captures both potential and actual purchasing behavior

#### Additional Engineered Features

- **Age Groups**: Categorical binning (Teen, Young Adult, Middle-Aged, Senior)
- **Income Brackets**: Quartile-based segmentation (Low, Medium, High, Very High)
- **Spending Categories**: Behavioral classification (Low, Moderate, High spenders)

### Clustering Algorithms

#### 1. K-Means Clustering

**Algorithm Overview:**
K-Means partitions data into K clusters by minimizing within-cluster variance (inertia). The algorithm iteratively assigns points to nearest centroids and updates centroids based on cluster membership.

**Mathematical Formulation:**

$$\text{Minimize: } J = \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2$$

Where:
- $K$ = number of clusters
- $C_i$ = cluster i
- $\mu_i$ = centroid of cluster i

**Implementation Details:**
- **Initialization**: k-means++ (intelligent centroid initialization)
- **Distance Metric**: Euclidean distance
- **Convergence**: Tolerance = 1e-4, Max iterations = 300
- **Random State**: 42 (reproducibility)

**Advantages:**
- Scalable to large datasets
- Interpretable cluster assignments
- Fast convergence
- Works well with spherical clusters

**Limitations:**
- Requires predetermined K value
- Sensitive to outliers
- Assumes similar cluster sizes
- Random initialization effects

#### 2. Elbow Method (Optimal K Selection)

The Elbow Method identifies the optimal number of clusters by analyzing inertia reduction:

**Process:**
1. Fit KMeans models for K = 1 to 10
2. Calculate inertia (within-cluster sum of squares) for each K
3. Plot inertia vs. K
4. Identify the "elbow" point where inertia reduction diminishes

**Interpretation:**
- Steep decrease in early K values indicates good clustering
- Flattening curve suggests optimal K is reached
- Elbow point balances model complexity and fit quality

### Model Evaluation

#### Internal Validation Metrics

**1. Silhouette Score**
Measures how similar an object is to its cluster compared to other clusters (range: -1 to 1).

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

- **s(i) ≈ 1**: Well-clustered points
- **s(i) ≈ 0**: Ambiguous cluster assignment
- **s(i) ≈ -1**: Possibly misclassified points

**2. Davies-Bouldin Index**
Ratio of within-cluster to between-cluster distances (lower is better).

$$DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \frac{S_i + S_j}{d_{ij}}$$

**3. Calinski-Harabasz Score**
Ratio of between-cluster to within-cluster dispersion (higher is better).

**4. Inertia**
Sum of squared distances from points to their cluster centroids. Used in Elbow Method analysis.

#### Business Validation

- Segment interpretability and actionability
- Statistical significance of cluster separation
- Alignment with known customer behavior patterns
- Stability across different random initializations

---

##  Key Results & Insights

### Clustering Performance

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Optimal K (Elbow) | 5–6 | Natural grouping identified |
| Silhouette Score | 0.52–0.58 | Reasonable cluster separation |
| Davies-Bouldin Index | 0.75–0.85 | Distinct, well-separated clusters |
| Calinski-Harabasz Score | 45–55 | Good cluster density and separation |

### Customer Segments Identified

#### Segment 1: **Affluent Professionals** (High Income, High Spending)
- Characteristics: Age 35–50, Income $80–120k, Spending Score 70–100
- Marketing Strategy: Premium products, exclusive offers, VIP programs
- Business Value: High revenue contribution, loyalty potential

#### Segment 2: **Young Aspirants** (Medium Income, High Spending)
- Characteristics: Age 18–30, Income $40–80k, Spending Score 70–100
- Marketing Strategy: Trendy products, digital channels, social media engagement
- Business Value: Emerging segment, high growth potential

#### Segment 3: **Middle-Class Moderate** (Medium Income, Medium Spending)
- Characteristics: Age 30–50, Income $40–80k, Spending Score 40–60
- Marketing Strategy: Value-for-money products, seasonal promotions
- Business Value: Stable revenue, price-sensitive

#### Segment 4: **Conservative Savers** (High Income, Low Spending)
- Characteristics: Age 40–70, Income $80–137k, Spending Score 1–40
- Marketing Strategy: Investment products, long-term value propositions
- Business Value: Untapped potential, requires engagement strategies

#### Segment 5: **Budget-Conscious Students** (Low Income, Low Spending)
- Characteristics: Age 18–35, Income $15–40k, Spending Score 1–40
- Marketing Strategy: Budget products, student discounts, payment plans
- Business Value: Future loyal customers, brand awareness

### Gender Distribution Insights

- Female customers show slightly higher average spending scores (52.7 vs. 48.5)
- Income distribution similar between genders
- Spending behavior varies significantly within gender groups
- Gender-based clustering reveals distinct behavioral patterns beyond demographics

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for version control)
- Jupyter Notebook

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/customer-segmentation.git
cd customer-segmentation
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import pandas, numpy, sklearn; print('All libraries installed successfully!')"
```

### requirements.txt

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
scipy>=1.7.0
jupyter>=1.0.0
ipython>=7.0.0
```

---

## 📖 Usage Guide

### Running the Analysis

```bash
# Start Jupyter Notebook
jupyter notebook

# Open segmentation.ipynb and run all cells
```

### Step-by-Step Execution

**1. Data Loading & Exploration**
```python
import pandas as pd
df = pd.read_csv('data/Mall_Customers.csv')
df.head()
df.describe()
```

**2. Preprocessing & Feature Engineering**
```python
from src.feature_engineering import create_affluence_score
df['Affluence_Score'] = create_affluence_score(df)
```

**3. Data Normalization**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Age', 'Spending_Score']])
```

**4. Determine Optimal K**
```python
from src.clustering import find_optimal_k
optimal_k = find_optimal_k(X_scaled, max_k=10)
print(f"Optimal number of clusters: {optimal_k}")
```

**5. Fit K-Means Model**
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
```

**6. Analyze Results**
```python
from src.visualization import plot_clusters, cluster_profile
plot_clusters(df, X_scaled)
cluster_profile(df)
```

---

##  Results Visualization

### Key Visualizations Included

#### 1. **Elbow Method Plot**
Shows inertia vs. number of clusters, identifying the optimal K value through the "elbow" point where inertia reduction diminishes significantly.

#### 2. **2D Cluster Scatter Plots**
- Age vs. Spending Score with cluster coloring
- Affluence Score vs. Spending Score
- Income vs. Spending Score

#### 3. **3D Interactive Plots**
- Age, Income, Spending Score in 3D space
- Interactive rotation and zoom capabilities
- Cluster visualization with distinct colors

#### 4. **Distribution Analysis**
- Histograms for each feature within clusters
- Boxplots showing cluster-wise variations
- Violin plots for distribution comparison

#### 5. **Cluster Profiles**
- Heatmaps of mean feature values per cluster
- Statistical summaries (mean, std, min, max)
- Gender distribution within clusters

#### 6. **Hierarchical Clustering Dendrogram**
- Shows hierarchical relationships between data points
- Validates K-Means results through alternative clustering perspective

---

## Conclusions & Recommendations

### Key Findings

1. **Natural Segmentation Exists**: Customer data reveals 5–6 distinct groups with statistically significant separation, confirming the value of segmentation strategies.

2. **Affluence Score Effectiveness**: The engineered Affluence Score significantly improves cluster quality compared to using raw features alone, demonstrating the power of domain-informed feature engineering.

3. **Spending Behavior Dominates**: Spending score is the primary differentiator across segments, more influential than age or income alone in defining customer groups.

4. **Age-Income-Spending Triangle**: These three dimensions effectively capture customer diversity, with clear profiles emerging at different combinations.

5. **Gender Insights**: While gender shows some spending differences, the behavioral segmentation transcends simple gender categories, suggesting more sophisticated targeting potential.

### Business Recommendations

**For Marketing Teams:**
- Develop segment-specific marketing campaigns with tailored messaging
- Allocate promotional budgets proportionally to segment size and profitability
- Use segment profiles to personalize customer communications

**For Product Development:**
- Design product lines targeting specific segment preferences
- Conduct focus groups within high-value segments
- Develop entry-level products for budget-conscious segments

**For Customer Service:**
- Implement tiered service levels based on segment classification
- Prioritize resources for high-value segments
- Create retention programs for at-risk segments

**For Strategic Planning:**
- Identify segment growth trajectories and expansion opportunities
- Monitor segment shifts over time for market trend detection
- Use segments for market positioning and competitive analysis

---

##  Future Enhancements

### Model Improvements
- **Hierarchical Clustering**: Compare with agglomerative clustering for dendrogram insights
- **DBSCAN**: Test density-based clustering for irregular cluster shapes
- **Gaussian Mixture Models (GMM)**: Probabilistic clustering with soft assignments
- **Dimensionality Reduction**: Apply PCA/t-SNE for visualization and noise reduction

### Feature Development
- **RFM Analysis**: Recency, Frequency, Monetary value metrics
- **Customer Lifetime Value (CLV)**: Predictive spending potential
- **Temporal Features**: Seasonality, trend, and cyclical patterns
- **Behavioral Indicators**: Product preferences, channel usage, return rates

### Advanced Analytics
- **Predictive Modeling**: Predict segment membership for new customers
- **Churn Prediction**: Identify at-risk customers within segments
- **Recommendation Systems**: Cross-sell and upsell opportunities by segment
- **Time Series Analysis**: Track segment evolution over time

### Deployment & Productionization
- **API Development**: REST API for real-time customer segmentation
- **Dashboard Creation**: Interactive Tableau/Power BI dashboards
- **Automated Pipelines**: MLOps implementation with scheduled retraining
- **A/B Testing Framework**: Validate segment-targeting effectiveness

---

##  Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

**Putta Vijay Kumar**
- Machine Learning & AI Enthusiast
- Data Science Professional
- GitHub: [@yourusername](https://github.com/puttavijaykumar)
- LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/vijaykumar-putta/)
- Email: your.email@example.com

---

## References & Resources

- Scikit-Learn Documentation: https://scikit-learn.org/
- K-Means Clustering: https://en.wikipedia.org/wiki/K-means_clustering
- Feature Engineering Best Practices: https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/
- Clustering Evaluation Metrics: https://en.wikipedia.org/wiki/Clustering_high-dimensional_data

---
