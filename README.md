 RBK Procurement Predictor — Full-Stack ML Web App (React + Flask)

This is a full-stack machine learning application to predict procurement amounts (₹) for Rythu Bharosa Kendras (RBKs) based on agricultural and geographic inputs like District, Mandal, Season, Quantity (MTs), and Number of Farmers.

 Tech Stack
- **Frontend**: React.js
- **Backend**: Flask (Python)
- **Modeling**: XGBoost or LightGBM (trained offline)
- **Explainability**: SHAP (global + local)
- **Reports**: Auto-generated PDF with plots & predictions
- **Maps**: Interactive choropleth using Plotly/Folium
- **Deployment**: Docker + CI/CD (GitHub Actions)

 Key Features
-  Predict procurement amount with confidence intervals
- SHAP explainability for each prediction
-  Choropleth map of predicted district-wise totals
-  Downloadable PDF report for any prediction
-  What-if simulator for sensitivity analysis
-  Dockerized deployment for portability
-  Planned monitoring and auto-retraining

###  Deliverables
- React UI with forms, visualizations, and download options
- Flask API for model prediction, SHAP, and PDF rendering
- Trained ML model (joblib) and preprocessing pipeline
- Clean code structure and automated CI pipeline
