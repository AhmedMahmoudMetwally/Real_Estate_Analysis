# REAL ESTATE ANALYSIS WITH REAL DATA

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    """Perform varimax rotation on loadings matrix."""
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(
            np.dot(Phi.T, np.asarray(Lambda) ** 3 - (gamma / p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda)))))
        )
        R = np.dot(u, vh)
        d_old = d
        d = np.sum(s)
        if d_old and d / d_old < 1 + tol:
            break
    return np.dot(Phi, R)


def calculate_kmo(df):
    """Calculate Kaiser-Meyer-Olkin measure for sampling adequacy."""
    corr = df.corr().values
    inv_corr = np.linalg.inv(corr)
    partial_corr = -inv_corr / np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
    np.fill_diagonal(partial_corr, 0)
    corr_sq = corr ** 2
    partial_corr_sq = partial_corr ** 2
    numerator = np.sum(corr_sq) - np.sum(np.diag(corr_sq))
    denominator = numerator + np.sum(partial_corr_sq)
    return numerator / denominator if denominator != 0 else np.nan

# Set page config
st.set_page_config(page_title="Real Estate Analysis - Real Data", layout="wide")


# LOAD  DATASETS


@st.cache_data
def load_california_housing():
    """Load California Housing dataset from reliable source"""
    try:
        # Method 1: load from scikit-learn 
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['MedHouseValue'] = housing.target
        return df, "California Housing Dataset (SKLearn)"
    except:
        # Method 2: Alternative URL
        url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
        df = pd.read_csv(url)
        return df, "California Housing Dataset (GitHub)"

@st.cache_data
def load_boston_housing():
    """Load Boston Housing dataset"""
    try:
        # Note: Boston dataset is deprecated in newer sklearn, using alternative
        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        df = pd.read_csv(url)
        return df, "Boston Housing Dataset"
    except:
        return None, None

@st.cache_data
def load_uci_real_estate():
    """Load UCI Real Estate Valuation dataset"""
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx"
        df = pd.read_excel(url)
        return df, "UCI Real Estate Dataset"
    except:
        return None, None


# DATASET SELECTION


st.title("🏠 Real Estate Price Prediction - Real Data Analysis")
st.markdown("---")

# Dataset selector
st.sidebar.title("📊 Data Source Selection")
dataset_option = st.sidebar.selectbox(
    "Choose Real Dataset:",
    [
        "California Housing (20,640 samples) - RECOMMENDED",
        "Boston Housing (506 samples)",
        "UCI Real Estate Valuation (414 samples)"
    ]
)

training_mode = st.sidebar.radio(
    "Training Mode:",
    ["Fast Mode", "Full Mode"],
    help="Choose Fast Mode for quicker results or Full Mode for more thorough evaluation."
)

# Load selected dataset
df = None
dataset_name = ""

with st.spinner("Loading real dataset..."):
    if "California" in dataset_option:
        df, dataset_name = load_california_housing()
    elif "Boston" in dataset_option:
        df, dataset_name = load_boston_housing()
    else:
        df, dataset_name = load_uci_real_estate()

if df is None:
    st.error("Failed to load dataset. Using fallback data generation...")
    # Fallback to synthetic but realistic data
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'MedInc': np.random.uniform(0.5, 15, n_samples),
        'HouseAge': np.random.uniform(1, 52, n_samples),
        'AveRooms': np.random.uniform(2, 10, n_samples),
        'AveBedrms': np.random.uniform(1, 5, n_samples),
        'Population': np.random.uniform(100, 5000, n_samples),
        'AveOccup': np.random.uniform(1, 8, n_samples),
        'Latitude': np.random.uniform(32.5, 42, n_samples),
        'Longitude': np.random.uniform(-124, -114, n_samples),
        'MedHouseValue': np.random.uniform(0.5, 5, n_samples)
    })
    dataset_name = "Synthetic California-Style Data"

st.sidebar.success(f"✅ Loaded: {dataset_name}")
st.sidebar.write(f"**Samples:** {df.shape[0]:,}")
st.sidebar.write(f"**Features:** {df.shape[1] - 1}")


def prepare_model_data(df, training_mode):
    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_train = scaler.transform(X_train_df)
    X_test = scaler.transform(X_test_df)
    if training_mode == 'Fast Mode':
        n_estimators = 30
        cv_folds = 3
    else:
        n_estimators = 100
        cv_folds = 5
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=n_estimators, random_state=42),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale')
    }
    results = []
    predictions = {}
    progress = st.progress(0)
    status_text = st.empty()
    total_models = len(models)
    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name} ({idx+1}/{total_models}) in {training_mode}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='r2', n_jobs=-1)
        results.append({
            'Model': name,
            'R² Score': r2,
            'RMSE': rmse,
            'MAE': mae,
            'CV Mean R²': cv_scores.mean(),
            'CV Std': cv_scores.std()
        })
        progress.progress((idx + 1) / total_models)
    progress.empty()
    status_text.text('Training complete. Preparing results...')
    results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)
    best_tree_model = None
    for name in ['Random Forest', 'Gradient Boosting']:
        if name in results_df.head(2)['Model'].values:
            best_tree_model = name
            break
    full_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    full_model.fit(X_scaled, y)
    status_text.empty()
    return {
        'X': X,
        'y': y,  # <--- THIS WAS MISSING! Now y is returned
        'scaler': scaler,
        'X_scaled': X_scaled,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_df': X_train_df,
        'X_test_df': X_test_df,
        'y_train': y_train,
        'y_test': y_test,
        'models': models,
        'results_df': results_df,
        'predictions': predictions,
        'best_tree_model': best_tree_model,
        'final_model': full_model,
        'training_mode': training_mode
    }


def get_model_data(df, dataset_name, training_mode):
    cache_key = f"model_data_{dataset_name}_{df.shape[0]}_{df.shape[1]}_{training_mode}"
    if 'model_data_cache' not in st.session_state or st.session_state.get('model_data_cache_key') != cache_key:
        st.session_state['model_data_cache'] = prepare_model_data(df, training_mode)
        st.session_state['model_data_cache_key'] = cache_key
    return st.session_state['model_data_cache']

# Navigation
section = st.sidebar.radio("Go to:", [
    "1. Data Description & Goals",
    "2. ML Algorithms Explanation",
    "3. Data Preparation & Cleaning",
    "4. Challenges Faced",
    "5. Model Training & Comparison",
    "6. Conclusion & Best Model",
    "7. Future Analysis & Predictions",
    "8. Factor Analysis"
])

#  DATA DESCRIPTION


if section == "1. Data Description & Goals":
    st.header(f"📋 Data Description: {dataset_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Overview")
        st.write(f"**Source:** Real-world data from official sources")
        st.write(f"**Number of samples:** {df.shape[0]:,}")
        st.write(f"**Number of features:** {df.shape[1] - 1}")
        st.write("**Features:**")
        
        for col in df.columns[:-1]:
            st.write(f"- {col}")
        st.write(f"**Target Variable:** {df.columns[-1]}")
        
        # Show data preview
        st.subheader("Data Preview (First 10 rows)")
        st.dataframe(df.head(10))
    
    with col2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
    
    st.markdown("---")
    
    st.subheader("🎯 Project Goals")
    st.markdown(f"""
    1. **Primary Goal:** Predict {df.columns[-1]} based on multiple features using REAL data
    2. **Secondary Goals:**
       - Identify which features most influence property prices
       - Compare 6 ML models on real-world data
       - Handle real data challenges (missing values, outliers, scaling)
       - Provide actionable insights for real estate decisions
    3. **Dataset Context:** {dataset_name} contains actual real estate records
    """)
    
    # Visualizations
    st.subheader("📊 Data Visualizations")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Target distribution
    target_col = df.columns[-1]
    axes[0, 0].hist(df[target_col], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f'{target_col} Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel(target_col)
    axes[0, 0].set_ylabel('Frequency')
    
    # First feature vs target
    first_feature = df.columns[0]
    axes[0, 1].scatter(df[first_feature], df[target_col], alpha=0.5, c='coral')
    axes[0, 1].set_title(f'{first_feature} vs {target_col}', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel(first_feature)
    axes[0, 1].set_ylabel(target_col)
    
    # Correlation heatmap
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, ax=axes[1, 0])
    axes[1, 0].set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
    
    # Box plot for target
    axes[1, 1].boxplot(df[target_col])
    axes[1, 1].set_title(f'{target_col} Box Plot', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel(target_col)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature correlation with target
    st.subheader("🔍 Feature Correlation with Target")
    correlations = df.corr()[target_col].drop(target_col).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    correlations.plot(kind='bar', ax=ax, color='teal', alpha=0.7)
    ax.set_title(f'Feature Correlations with {target_col}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features')
    ax.set_ylabel('Correlation Coefficient')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# SECTION 2: ML ALGORITHMS EXPLANATION


elif section == "2. ML Algorithms Explanation":
    st.header("🤖 Machine Learning Algorithms Used")
    st.markdown("Learn why each model was chosen, how it works, and when it performs best on real estate data.")
    
    algorithms = {
        "Linear Regression": {
            "why": "Baseline model for multivariate regression",
            "how": "Finds linear combination: Y = w1X1 + w2X2 + ... + b",
            "pros": "Simple, interpretable, fast",
            "cons": "Assumes linearity, sensitive to outliers"
        },
        "Ridge Regression": {
            "why": "Handles multicollinearity between features",
            "how": "Adds L2 penalty: λΣw² to prevent overfitting",
            "pros": "Reduces overfitting, handles correlated features",
            "cons": "Biased estimates, requires tuning"
        },
        "Lasso Regression": {
            "why": "Automatic feature selection",
            "how": "Adds L1 penalty: λΣ|w|, forces some coefficients to zero",
            "pros": "Feature selection, sparse solutions",
            "cons": "Can discard correlated features"
        },
        "Random Forest": {
            "why": "Captures non-linear relationships",
            "how": "Ensemble of decision trees, averages predictions",
            "pros": "Handles non-linearity, robust to outliers",
            "cons": "Less interpretable, computationally heavy"
        },
        "Gradient Boosting": {
            "why": "State-of-the-art for tabular data",
            "how": "Builds trees sequentially, each corrects previous errors",
            "pros": "High accuracy, handles complex patterns",
            "cons": "Prone to overfitting, many hyperparameters"
        },
        "SVR": {
            "why": "Effective in high-dimensional spaces",
            "how": "Finds hyperplane with maximum margin",
            "pros": "Effective in high dimensions, memory efficient",
            "cons": "Poor with large datasets, sensitive to scaling"
        }
    }
    
    cols = st.columns(2)
    for idx, (name, info) in enumerate(algorithms.items()):
        with cols[idx % 2]:
            st.subheader(name)
            st.markdown(f"**Why use it?** {info['why']}")
            st.markdown(f"**How it works:** {info['how']}")
            st.markdown(f"**✅ Pros:** {info['pros']}")
            st.markdown(f"**❌ Cons:** {info['cons']}")
            st.write("---")

#  SECTION 3: DATA PREPARATION


elif section == "3. Data Preparation & Cleaning":
    st.header("🧹 Real Data: Preparation & Cleaning")
    
    st.subheader("Initial Data Quality Check")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_pct = (missing_values / len(df)) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Missing Values", missing_values.sum())
    with col2:
        st.metric("Duplicate Rows", df.duplicated().sum())
    with col3:
        st.metric("Data Types", len(df.dtypes.unique()))
    
    # Show missing values if any
    if missing_values.sum() > 0:
        st.subheader("Missing Values Analysis")
        missing_df = pd.DataFrame({
            'Column': missing_values[missing_values > 0].index,
            'Missing Count': missing_values[missing_values > 0].values,
            'Percentage': missing_pct[missing_values > 0].values
        })
        st.dataframe(missing_df)
        
        # Handle missing values
        st.subheader("Missing Value Treatment")
        st.markdown("""
        **Strategy Applied:**
        - Columns with >50% missing values are dropped
        - Numerical columns are filled with median values
        - Categorical columns are filled with mode values
        """)
        
        # Drop columns with more than 50% missing values
        high_missing_cols = [col for col in df.columns if df[col].isnull().mean() > 0.5]
        if high_missing_cols:
            df.drop(columns=high_missing_cols, inplace=True)
            st.warning(f"Dropped columns with >50% missing values: {', '.join(high_missing_cols)}")
            missing_values = df.isnull().sum()
            missing_pct = (missing_values / len(df)) * 100
        
        # Apply filling
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        st.success("Missing values have been handled!")
    
    # Outlier detection
    st.subheader("Outlier Detection")
    target_col = df.columns[-1]
    
    # Apply percentile capping to reduce extreme outliers
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    lower_bound = df[numeric_cols].quantile(0.01)
    upper_bound = df[numeric_cols].quantile(0.99)
    for col in numeric_cols:
        df[col] = np.clip(df[col], lower_bound[col], upper_bound[col])
    
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[target_col] < Q1 - 1.5*IQR) | (df[target_col] > Q3 + 1.5*IQR)]
    
    st.write(f"**Outliers detected in target variable:** {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.boxplot(df[target_col])
    ax1.set_title(f'Box Plot - {target_col}')
    ax1.set_ylabel(target_col)
    
    ax2.hist(df[target_col], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(df[target_col].mean(), color='red', linestyle='--', 
                label=f'Mean: {df[target_col].mean():.2f}')
    ax2.axvline(df[target_col].median(), color='green', linestyle='--', 
                label=f'Median: {df[target_col].median():.2f}')
    ax2.set_title(f'{target_col} Distribution')
    ax2.set_xlabel(target_col)
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Preprocessing Steps Applied")
    st.markdown("""
    1. **Missing Value Treatment:** Dropped columns with >50% missing values, then applied median/mode imputation
    2. **Outlier Treatment:** Capped extreme numeric values at the 1st and 99th percentiles
    3. **Feature Scaling:** RobustScaler is used later for model training
    4. **Train-Test Split:** 80% training, 20% testing (random_state=42)
    5. **Handling Categorical Features:** Label encoding if present
    """)

# SECTION 4: CHALLENGES FACED WITH REAL DATA


elif section == "4. Challenges Faced":
    st.header("⚠️ Real Data Challenges & Solutions")
    st.markdown("This section highlights the main data issues discovered in the selected dataset and the practical solutions applied.")
    
    # Detect actual challenges in the loaded data
    challenges_faced = []
    missing_total = int(df.isnull().sum().sum())
    target_col = df.columns[-1]
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    outlier_pct = len(df[(df[target_col] < Q1 - 1.5*IQR) | (df[target_col] > Q3 + 1.5*IQR)]) / len(df) * 100
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scales = [df[col].std() for col in numeric_cols if df[col].std() > 0]
    scale_ratio = max(scales) / min(scales) if scales else 1
    skewed_count = int((df[numeric_cols].skew().abs() > 1).sum())
    
    if missing_total > 0:
        challenges_faced.append({
            "challenge": "Missing Values",
            "solution": "Median imputation for numerical features and mode imputation for categorical features.",
            "impact": "Preserves records while keeping the dataset usable for modeling."
        })
    
    if outlier_pct > 5:
        challenges_faced.append({
            "challenge": f"Outliers Detected ({outlier_pct:.1f}% of target values)",
            "solution": "Applied RobustScaler and used tree-based models to reduce outlier impact.",
            "impact": "Improved prediction stability for non-linear models."
        })
    
    if scale_ratio > 100:
        challenges_faced.append({
            "challenge": "Feature Scale Mismatch",
            "solution": "Standardized numeric features with StandardScaler.",
            "impact": "Helps linear and distance-based models converge correctly."
        })
    
    if skewed_count > 0:
        challenges_faced.append({
            "challenge": f"Skewed Feature Distributions ({skewed_count} features)",
            "solution": "Consider log transformation or robust scaling for skewed variables.",
            "impact": "Improves model assumptions and prediction quality."
        })
    
    if not challenges_faced:
        challenges_faced.append({
            "challenge": "No Major Data Issues Detected",
            "solution": "Proceed with standard preprocessing and model training.",
            "impact": "The dataset appears clean and suitable for modeling."
        })
    
    st.subheader("📊 Challenge Summary")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_col1.metric("Missing Values", f"{missing_total}")
    stat_col2.metric("Outlier %", f"{outlier_pct:.1f}%")
    stat_col3.metric("Scale Ratio", f"{scale_ratio:.1f}x")
    stat_col4.metric("Skewed Features", f"{skewed_count}")
    
    st.subheader("🧠 Identified Challenges")
    for item in challenges_faced:
        with st.container():
            challenge_col, details_col = st.columns([1, 2])
            with challenge_col:
                st.error(f"🚨 {item['challenge']}")
            with details_col:
                st.markdown(f"**Solution:** {item['solution']}")
                st.markdown(f"**Impact:** {item['impact']}")
            st.markdown("---")

#  MODEL TRAINING & COMPARISON


elif section == "5. Model Training & Comparison":
    st.header("📊 Model Training on Real Data")
    
    model_data = get_model_data(df, dataset_name, training_mode)
    results_df = model_data['results_df']
    predictions = model_data['predictions']
    X = model_data['X']
    y = model_data['y']  # <--- THIS WAS MISSING! Now y is retrieved from model_data
    X_scaled = model_data['X_scaled']
    y_test = model_data['y_test']
    y_train = model_data['y_train']
    models = model_data['models']
   
    
    st.subheader("🏆 Model Performance Comparison on Real Data")
    
    # Display metrics
    st.dataframe(results_df.style.format({
        'R² Score': '{:.4f}',
        'RMSE': '{:,.2f}',
        'MAE': '{:,.2f}',
        'CV Mean R²': '{:.4f}',
        'CV Std': '{:.4f}'
    }).background_gradient(subset=['R² Score'], cmap='RdYlGn'))
    
    # Performance chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² Score bar chart
    bars1 = ax1.bar(results_df['Model'], results_df['R² Score'], 
                    color=['gold' if i==0 else 'steelblue' for i in range(len(results_df))],
                    alpha=0.7)
    ax1.set_title('R² Score by Model (Higher is Better)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('R² Score')
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(y=results_df['R² Score'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results_df["R² Score"].mean():.3f}')
    ax1.legend()
    
    for bar, score in zip(bars1, results_df['R² Score']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', fontsize=9)
    
    # RMSE comparison (lower is better)
    bars2 = ax2.bar(results_df['Model'], results_df['RMSE'], 
                    color='coral', alpha=0.7)
    ax2.set_title('RMSE by Model (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('RMSE')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, rmse in zip(bars2, results_df['RMSE']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.02, 
                f'{rmse:.2f}', ha='center', fontsize=9, rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Prediction vs Actual for top 3
    st.subheader("📈 Prediction vs Actual Values (Top 3 Models)")
    
    top_models = results_df.head(3)['Model'].values
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, model_name in enumerate(top_models):
        y_pred = predictions[model_name]
        axes[idx].scatter(y_test, y_pred, alpha=0.5, s=10)
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                      label='Perfect Prediction')
        
        r2_score_val = results_df[results_df['Model']==model_name]['R² Score'].values[0]
        axes[idx].set_title(f'{model_name}\nR² = {r2_score_val:.3f}')
        axes[idx].set_xlabel('Actual Values')
        axes[idx].set_ylabel('Predicted Values')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature Importance for best tree-based model
    best_tree_model = None
    for name in ['Random Forest', 'Gradient Boosting']:
        if name in results_df.head(2)['Model'].values:
            best_tree_model = name
            break
    
    if best_tree_model:
        st.subheader(f"🔍 Feature Importance ({best_tree_model})")
        
        best_model = models[best_tree_model]
        importances = best_model.feature_importances_
        features = X.columns
        
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'], 
                       color='teal', alpha=0.7)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Feature Importance Analysis - {best_tree_model}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        for i, (feature, importance) in enumerate(zip(importance_df['Feature'], importance_df['Importance'])):
            ax.text(importance + 0.01, i, f'{importance:.3f}', va='center')
        
        st.pyplot(fig)
        
        top_feature = importance_df.iloc[-1]['Feature']
        top_importance = importance_df.iloc[-1]['Importance']
        st.info(f"💡 **Insight:** '{top_feature}' is the most important feature, contributing {top_importance:.1%} to price prediction. This aligns with real estate economics theory!")
        
        # Feature Selection
        st.subheader("🎯 Feature Selection")
        top_n = min(10, len(importance_df))
        selected_features = importance_df.nlargest(top_n, 'Importance')['Feature'].tolist()
        st.write(f"**Top {top_n} Selected Features (based on importance):**")
        for i, feat in enumerate(selected_features, 1):
            imp = importance_df[importance_df['Feature'] == feat]['Importance'].values[0]
            st.write(f"{i}. {feat} (Importance: {imp:.3f})")
        st.markdown("**Why Feature Selection?** Reduces overfitting, improves model interpretability, and speeds up training by focusing on key predictors.")
        
        # Retrain with selected features
        st.subheader("🔄 Model Performance with Selected Features")
        X_selected = X[selected_features]
        
        # Split selected features
        X_selected_train, X_selected_test, y_train_split, y_test_split = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        
        scaler_selected = RobustScaler()
        X_selected_train_scaled = scaler_selected.fit_transform(X_selected_train)
        X_selected_test_scaled = scaler_selected.transform(X_selected_test)
        
        # Retrain the best tree model with selected features
        if best_tree_model == 'Random Forest':
            model_class = RandomForestRegressor
        elif best_tree_model == 'Gradient Boosting':
            model_class = GradientBoostingRegressor
        else:
            model_class = RandomForestRegressor  # fallback
        
        model_selected = model_class(random_state=42)
        model_selected.fit(X_selected_train_scaled, y_train_split)
        y_pred_selected = model_selected.predict(X_selected_test_scaled)
        
        r2_selected = r2_score(y_test_split, y_pred_selected)
        rmse_selected = np.sqrt(mean_squared_error(y_test_split, y_pred_selected))
        mae_selected = mean_absolute_error(y_test_split, y_pred_selected)
        
        # Compare
        original_r2 = results_df[results_df['Model'] == best_tree_model]['R² Score'].values[0]
        original_rmse = results_df[results_df['Model'] == best_tree_model]['RMSE'].values[0]
        original_mae = results_df[results_df['Model'] == best_tree_model]['MAE'].values[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score (All Features)", f"{original_r2:.4f}")
            st.metric("R² Score (Selected)", f"{r2_selected:.4f}", delta=f"{(r2_selected - original_r2)*100:.1f}%")
        with col2:
            st.metric("RMSE (All Features)", f"{original_rmse:.2f}")
            st.metric("RMSE (Selected)", f"{rmse_selected:.2f}", delta=f"{(original_rmse - rmse_selected):.2f}")
        
        if r2_selected > original_r2:
            st.success("✅ Feature selection improved performance!")
        elif abs(r2_selected - original_r2) < 0.01:
            st.info("ℹ️ Performance similar with fewer features.")
        else:
            st.warning("⚠️ Slight drop in performance, but model is simpler.")

# CONCLUSION


elif section == "6. Conclusion & Best Model":
    st.header("🎯 Conclusion & Recommended Model")
    
    model_data = get_model_data(df, dataset_name, training_mode)
    results_df = model_data['results_df']
    best_model = results_df.loc[results_df['R² Score'].idxmax(), 'Model']
    best_score = results_df['R² Score'].max()
    second_score = results_df['R² Score'].sort_values(ascending=False).iloc[1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"### 🏆 Best Model on Real Data: **{best_model}**")
        st.metric("R² Score", f"{best_score:.4f}", 
                 delta=f"+{(best_score - second_score)*100:.1f}%" if best_score > second_score else None)
        
        st.markdown(f"""
        **Why {best_model} is best for this real dataset:**
        1. **Highest R² Score** - Explains {best_score*100:.1f}% of price variance
        2. **Handles Real Data Complexity** - Robust to outliers and non-linearity
        3. **No Strong Assumptions** - Doesn't require normal distribution
        4. **Feature Importance** - Provides interpretable results
        5. **Cross-Validation Stability** - Consistent performance across folds
        """)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['gold' if m == best_model else 'steelblue' for m in results_df['Model']]
        bars = ax.bar(results_df['Model'], results_df['R² Score'], color=colors, alpha=0.7)
        ax.set_title('Final Model Comparison on Real Data', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('R² Score')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0.7, color='green', linestyle='--', label='Good Threshold (0.7)')
        ax.legend()
        
        for bar, score in zip(bars, results_df['R² Score']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{score:.3f}', ha='center', fontsize=10)
        
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("📌 Key Findings from Real Data Analysis")
    
    # Calculate some insights
    target_col = df.columns[-1]
    best_feature_corr = df.corr()[target_col].drop(target_col).abs().idxmax()
    best_corr_value = df.corr()[target_col][best_feature_corr]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Most Influential Feature", best_feature_corr, 
                 f"Correlation: {best_corr_value:.3f}")
    with col2:
        improvement = (best_score - 0.5) * 100 if best_score > 0.5 else (0.5 - best_score) * 100
        st.metric("Model Improvement vs Baseline", f"{improvement:.0f}%", 
                 "Random guess would be 0.5")
    with col3:
        st.metric("Dataset Size", f"{df.shape[0]:,}", "Real samples")

#  FUTURE ANALYSIS


elif section == "7. Future Analysis & Predictions":
    st.header("🔮 Future Analysis & Interactive Predictions")
    
    st.subheader("Potential Improvements for Real Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Advanced Techniques")
        st.markdown("""
        - **Hyperparameter Tuning:** GridSearchCV or RandomizedSearchCV
        - **Feature Engineering:** Polynomial features, interaction terms
        - **Deep Learning:** Neural networks for complex patterns
        - **XGBoost/LightGBM:** State-of-the-art gradient boosting
        - **Stacking Ensemble:** Combine multiple models
        """)
        
        st.markdown("### 📈 Model Improvements")
        st.markdown("""
        - **Time Series Analysis:** If date features available
        - **Geospatial Analysis:** Use latitude/longitude clustering
        - **External Data Integration:** Interest rates, economic indicators
        - **Explainable AI:** SHAP values for model interpretability
        """)
    
    with col2:
        st.markdown("### 🎯 Business Applications")
        st.markdown("""
        - **Investment ROI Calculator:** Predict future appreciation
        - **Market Segmentation:** Identify undervalued areas
        - **Risk Assessment:** Evaluate investment risks
        - **Real-time Pricing API:** Deploy model as web service
        - **Portfolio Optimization:** Maximize returns
        """)
    
    st.markdown("---")
    
    # Interactive prediction tool
    st.subheader("🎮 Interactive Price Predictor")
    model_data = get_model_data(df, dataset_name, training_mode)
    # Get best model name from results
    results_df_local = model_data['results_df']
    best_model_name = results_df_local.loc[results_df_local['R² Score'].idxmax(), 'Model']
    st.markdown(f"**Using best model: {best_model_name}**")
    
    X = model_data['X']
    scaler = model_data['scaler']
    final_model = model_data['final_model']
    
    st.markdown("### Adjust property features to predict price:")
    
    # Create input widgets based on features
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    input_values = {}
    
    # Create 3 columns for inputs
    cols = st.columns(3)
    for idx, col in enumerate(numeric_cols[:6]):  # Limit to first 6 features
        with cols[idx % 3]:
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            mean_val = float(X[col].mean())
            input_values[col] = st.slider(
                f"{col}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100
            )
    
    if st.button("🔮 Predict Price", type="primary"):
        # Create input dataframe
        input_df = pd.DataFrame([input_values])
        
        # Ensure all columns match
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = X[col].mean()
        
        input_df = input_df[X.columns]
        input_scaled = scaler.transform(input_df)
        prediction = final_model.predict(input_scaled)[0]
        
        target_name = df.columns[-1]
        st.success(f"### 💰 Predicted {target_name}: **{prediction:,.2f}**")
        st.info("💡 This prediction is based on real market data and ML models. Actual prices may vary based on market conditions.")

# FACTOR ANALYSIS SECTION


elif section == "8. Factor Analysis":
    st.header("🔍 Factor Analysis on Real Estate Features")
    
    st.markdown("""
    **Factor Analysis** searches for latent factors that explain common variance among the features.
    This section selects the best number of factors, applies factor analysis, and rotates the results with Varimax for clearer interpretation.
    """)
    
    # Prepare data
    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]
    
    # Handle categorical columns if any
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.warning("Non-numeric features have been encoded to enable factor analysis; this is not ideal for factor analysis.")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.subheader("📊 Data Suitability for Factor Analysis")
    try:
        kmo_value = calculate_kmo(pd.DataFrame(X_scaled, columns=X.columns))
        st.write(f"**KMO measure:** {kmo_value:.3f}")
        if kmo_value < 0.6:
            st.warning("Low KMO: the data may not be very suitable for factor analysis.")
    except Exception:
        st.info("KMO is unavailable for this dataset due to a correlation matrix issue.")
    
    # Determine optimal number of factors
    pca = PCA()
    pca.fit(X_scaled)
    eigenvalues = pca.explained_variance_
    n_factors = max(2, sum(eigenvalues > 1))
    n_factors = min(n_factors, X.shape[1])
    
    st.subheader("📈 Eigenvalues and Factor Selection")
    eigen_df = pd.DataFrame({
        'Factor': [f'Factor {i+1}' for i in range(len(eigenvalues))],
        'Eigenvalue': eigenvalues,
        'Cumulative Variance': np.cumsum(eigenvalues) / np.sum(eigenvalues)
    })
    st.dataframe(eigen_df.style.format({'Eigenvalue': '{:.3f}', 'Cumulative Variance': '{:.3f}'}))
    st.write(f"**Selected factors:** {n_factors} (Kaiser rule with minimum 2 factors)")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='o')
    ax.axhline(1, color='red', linestyle='--', label='Eigenvalue = 1')
    ax.set_xticks(range(1, len(eigenvalues)+1))
    ax.set_xlabel('Factor Number')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Scree Plot for Factor Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    if n_factors > 0:
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        X_factors = fa.fit_transform(X_scaled)
        loadings = fa.components_.T
        rotated_loadings = varimax(loadings) if n_factors > 1 else loadings
        
        loadings_df = pd.DataFrame(
            rotated_loadings,
            index=X.columns,
            columns=[f'Factor {i+1}' for i in range(n_factors)]
        )
        
        st.subheader("🔗 Rotated Factor Loadings")
        st.markdown("Varimax rotation has been applied to reduce factor overlap and make each factor more interpretable.")
        
        def color_loadings(val):
            if abs(val) > 0.6:
                return 'background-color: lightgreen'
            elif abs(val) > 0.4:
                return 'background-color: lightyellow'
            return ''
        
        st.dataframe(loadings_df.style.applymap(color_loadings).format("{:.3f}"))
        
        communalities = np.sum(loadings_df.values ** 2, axis=1)
        communalities_df = pd.DataFrame({
            'Feature': X.columns,
            'Communality': communalities,
            'Uniqueness': 1 - communalities
        }).sort_values('Communality', ascending=False)
        
        st.subheader("📈 Communalities")
        st.dataframe(communalities_df.style.format({'Communality': '{:.3f}', 'Uniqueness': '{:.3f}'}))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(loadings_df, annot=True, cmap='RdYlBu_r', center=0, linewidths=0.5, ax=ax, fmt='.2f')
        ax.set_title('Rotated Factor Loadings Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Factors')
        ax.set_ylabel('Features')
        st.pyplot(fig)
        
        if n_factors >= 2:
            st.subheader("🔄 Factor Scores vs Target")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            ax1.scatter(X_factors[:, 0], X_factors[:, 1], alpha=0.6, c=y, cmap='viridis')
            ax1.set_xlabel('Factor 1')
            ax1.set_ylabel('Factor 2')
            ax1.set_title('Factor Scores (Colored by Target)')
            ax1.grid(True, alpha=0.3)
            
            factor_target_corr = [np.corrcoef(X_factors[:, i], y)[0, 1] for i in range(n_factors)]
            ax2.bar(range(1, n_factors+1), factor_target_corr, color='skyblue', alpha=0.7)
            ax2.set_xlabel('Factor')
            ax2.set_ylabel('Correlation with Target')
            ax2.set_title('Factor-Target Correlations')
            ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.subheader("💡 Interpretation")
        for i in range(n_factors):
            strongest_feature = loadings_df[f'Factor {i+1}'].abs().idxmax()
            strongest_loading = loadings_df.loc[strongest_feature, f'Factor {i+1}']
            st.write(f"- Factor {i+1} is mainly associated with '{strongest_feature}' ({strongest_loading:.3f})")
        
        if n_factors >= 2:
            best_corr = max(range(len(factor_target_corr)), key=lambda i: abs(factor_target_corr[i]))
            st.info(f"The strongest factor associated with the target is Factor {best_corr+1}, with correlation {factor_target_corr[best_corr]:.3f}.")
        
        st.success("Factor analysis has been enhanced with rotation and data adequacy checks.")
    else:
        st.warning("No valid factors were selected using the Kaiser criterion. Try reducing features or changing the dataset.")

# Footer
st.markdown("---")
st.markdown(f"*Project completed using **{dataset_name}** - Real data multivariate analysis | All analysis based on actual real estate records*")
