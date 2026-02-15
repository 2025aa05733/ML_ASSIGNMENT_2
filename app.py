# Streamlit App for ML Assignment 2
# Student Performance Prediction

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report

# page config
st.set_page_config(page_title="Student Performance - ML Assignment 2", page_icon="📚", layout="wide")

# load models
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_model.pkl',
        'Decision Tree': 'dt_model.pkl',
        'KNN': 'knn_model.pkl',
        'Naive Bayes': 'nb_model.pkl',
        'Random Forest': 'rf_model.pkl',
        'XGBoost': 'xgb_model.pkl'
    }
    for name, file in model_files.items():
        path = f'models/{file}'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    return models

@st.cache_resource
def load_preprocessors():
    path = 'models/preprocessors.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_results():
    path = 'models/model_comparison.csv'
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None

# preprocess uploaded data
def preprocess_uploaded_data(df, preprocessors):
    X = df.copy()
    
    # check for target column
    y = None
    y_enc = None
    if 'G3' in X.columns:
        y = X['G3']
        X = X.drop('G3', axis=1)
    
    label_encoders = preprocessors['label_encoders']
    scaler = preprocessors['scaler']
    target_enc = preprocessors['target_encoder']
    
    # encode categorical cols
    cat_cols = [c for c in X.columns if c in label_encoders]
    for col in cat_cols:
        le = label_encoders[col]
        X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    
    # scale numerical cols
    num_cols = [c for c in X.columns if c not in cat_cols]
    if scaler and len(num_cols) > 0:
        X[num_cols] = scaler.transform(X[num_cols])
    
    # encode target
    if y is not None and target_enc:
        y_enc = y.apply(lambda x: target_enc.transform([x])[0] if x in target_enc.classes_ else 0)
    
    return X, y, y_enc, target_enc

# main app
def main():
    st.title("📚 Student Performance Prediction")
    st.write("### BITS M.Tech AIML - ML Assignment 2")
    st.markdown("---")
    
    # load everything
    models = load_models()
    preprocessors = load_preprocessors()
    results_df = load_results()
    
    if not models:
        st.error("Models not found! Run train_models.py first.")
        return
    
    # sidebar - model selection
    st.sidebar.header("Settings")
    selected_model = st.sidebar.selectbox("Choose Model", list(models.keys()), index=5)
    
    if results_df is not None:
        st.sidebar.markdown("---")
        st.sidebar.write("### Quick Stats")
        st.sidebar.dataframe(results_df.round(4))
    
    # tabs
    tab1, tab2, tab3 = st.tabs(["Upload & Predict", "Compare Models", "About"])
    
    # Tab 1 - Upload and predict
    with tab1:
        st.write("### Upload Test Data")
        st.info("Upload a CSV file with student data. If G3 column exists, metrics will be calculated.")
        
        uploaded = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")
                
                with st.expander("Preview Data"):
                    st.dataframe(df.head(10))
                
                if preprocessors:
                    X, y_true, y_enc, target_enc = preprocess_uploaded_data(df, preprocessors)
                    
                    model = models[selected_model]
                    y_pred_enc = model.predict(X)
                    
                    if target_enc:
                        y_pred = target_enc.inverse_transform(y_pred_enc)
                    else:
                        y_pred = y_pred_enc
                    
                    # show predictions
                    st.write("### Predictions")
                    result_df = df.copy()
                    result_df['Predicted_G3'] = y_pred
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.dataframe(result_df[['Predicted_G3'] + list(df.columns[:5])].head(15))
                    
                    with col2:
                        fig, ax = plt.subplots()
                        pd.Series(y_pred).value_counts().sort_index().plot(kind='bar', ax=ax, color='teal')
                        ax.set_xlabel('Grade')
                        ax.set_ylabel('Count')
                        ax.set_title('Prediction Distribution')
                        st.pyplot(fig)
                    
                    # if actual labels exist, show metrics
                    if y_true is not None:
                        st.write("### Metrics")
                        
                        acc = accuracy_score(y_true, y_pred)
                        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
                        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
                        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                        mcc = matthews_corrcoef(y_true, y_pred)
                        
                        try:
                            y_proba = model.predict_proba(X)
                            auc = roc_auc_score(y_enc, y_proba, multi_class='ovo', average='macro')
                        except:
                            auc = None
                        
                        c1, c2, c3, c4, c5, c6 = st.columns(6)
                        c1.metric("Accuracy", f"{acc:.4f}")
                        c2.metric("AUC", f"{auc:.4f}" if auc else "N/A")
                        c3.metric("Precision", f"{prec:.4f}")
                        c4.metric("Recall", f"{rec:.4f}")
                        c5.metric("F1", f"{f1:.4f}")
                        c6.metric("MCC", f"{mcc:.4f}")
                        
                        # confusion matrix
                        st.write("### Confusion Matrix")
                        cm = confusion_matrix(y_true, y_pred)
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)
                        
                        # classification report
                        st.write("### Classification Report")
                        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                        st.dataframe(pd.DataFrame(report).T.round(4))
                    
                    # download button
                    st.write("### Download")
                    csv = result_df.to_csv(index=False)
                    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
                    
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Tab 2 - Model comparison
    with tab2:
        st.write("### Model Comparison")
        
        if results_df is not None:
            st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
            
            st.write("### Charts")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots()
                results_df['Accuracy'].plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title('Accuracy')
                ax.set_ylabel('Score')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots()
                results_df['AUC'].plot(kind='bar', ax=ax, color='coral')
                ax.set_title('AUC Score')
                ax.set_ylabel('Score')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            # all metrics chart
            st.write("### All Metrics")
            fig, ax = plt.subplots(figsize=(12, 5))
            results_df.plot(kind='bar', ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.02, 1))
            plt.tight_layout()
            st.pyplot(fig)
            
            # observations
            st.write("### Model Observations")
            obs = {
                'Logistic Regression': 'Good AUC but lower accuracy. Linear model struggles with complex patterns.',
                'Decision Tree': 'Decent accuracy but can overfit. Easy to interpret.',
                'KNN': 'Lower performance. Sensitive to scaling and number of neighbors.',
                'Naive Bayes': 'Fast but assumes independence. Lowest accuracy here.',
                'Random Forest': 'Best AUC score. Ensemble of trees reduces overfitting.',
                'XGBoost': 'Best overall. Gradient boosting handles imbalanced classes well.'
            }
            for model, observation in obs.items():
                st.write(f"**{model}**: {observation}")
        else:
            st.warning("No results found. Run train_models.py first.")
    
    # Tab 3 - About
    with tab3:
        st.write("### About")
        st.write("""
        **Problem**: Predict student final grades (G3) based on various features.
        
        **Dataset**: Student Performance Dataset from UCI Repository
        - 649 students
        - 32 features (demographic, family, academic, lifestyle)
        - Target: G3 (final grade 0-20)
        
        **Models Used**:
        1. Logistic Regression
        2. Decision Tree
        3. K-Nearest Neighbors
        4. Naive Bayes (Gaussian)
        5. Random Forest
        6. XGBoost
        
        **Metrics**:
        - Accuracy, AUC (OvO), Precision, Recall, F1, MCC
        
        ---
        BITS M.Tech AIML - ML Assignment 2
        """)
    
    st.markdown("---")
    st.caption("ML Assignment 2 | Student Performance Prediction")

if __name__ == "__main__":
    main()
