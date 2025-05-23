# Comparative-Analysis-of-Machine-Learning-Algorithms-for-Diabetes-Prediction

## Get Started
Jump into predicting diabetes onset with the Pima Indians Diabetes Database! This project uses 768 records with 8 features (like Glucose and BMI) to classify diabetes cases (35% positive). You'll train and compare four models—Logistic Regression, Decision Tree, Random Forest, and XGBoost—optimized to handle missing data, class imbalance, and overfitting. XGBoost shines with a Recall of 0.82 and F1-Score of 0.68, perfect for catching critical cases.

## What You'll Achieve
- **Clean Data**: Use KNN imputation and SMOTE to preprocess the dataset.
- **Optimize Models**: Tune hyperparameters to balance performance.
- **Interpret Results**: Discover Glucose as the top predictor using SHAP.
- **Compare Performance**: Evaluate models on accuracy, precision, recall, F1-score, and AUC.

## Set Up Your Environment
Get the project running in minutes:
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/[your-username]/diabetes-prediction.git
   cd diabetes-prediction
   ```
2. **Install Dependencies**:

Ensure Python 3.8+ is installed. The requirements include:
   - pandas
   - numpy
   - scikit-learn
   - xgboost
   - matplotlib
   - seaborn
   - shap
   - imbalanced-learn

3. **Download the Dataset**:
   Grab the Pima Indians Diabetes Database from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it in `data/pima-indians-diabetes.csv`.


## Understand the Results
The models were trained on a 70% train, 15% cross-validation, and 15% test split. Check out the test set performance:

| Model              | Accuracy | Precision | Recall | F1-Score | AUC  |
|--------------------|----------|-----------|--------|----------|------|
| Logistic Regression| 0.74     | 0.73      | 0.50   | 0.59     | 0.85 |
| Decision Tree      | 0.75     | 0.80      | 0.45   | 0.58     | 0.82 |
| XGBoost            | 0.71     | 0.58      | 0.82   | 0.68     | 0.81 |
| Random Forest      | 0.77     | 0.79      | 0.52   | 0.63     | 0.85 |

**Take Action**:
- **Choose XGBoost** for medical applications where high recall (0.82) minimizes missed diabetes cases.
- **Use Random Forest** for balanced accuracy (0.77) and strong AUC (0.85).
- **Analyze Feature Importance**: Run the notebook to visualize SHAP plots, confirming Glucose as the key driver.

## Compare with Research
See how this project stacks up against published studies on the same dataset:

| Paper | Model | Accuracy | AUC | Why We're Different |
|-------|-------|----------|-----|---------------------|
| Hennebelle et al. (2023) | Random Forest | 0.7827 | - | Their feature selection boosts accuracy slightly; our RF (0.77) uses all features for robustness. |
| Hennebelle et al. (2023) | Logistic Regression | 0.7227 | - | Our LR (0.74) wins with SMOTE and tuned regularization. |
| Mousa et al. (2023) | LSTM | 0.85 | 0.89 | LSTM is powerful but complex; our models are simpler and interpretable. |
| Mousa et al. (2023) | Random Forest | 0.78 | 0.81 | Our RF (AUC=0.85) outperforms theirs, thanks to SMOTE. |
| Mousa et al. (2023) | CNN | 0.82 | 0.86 | Our XGBoost prioritizes recall (0.68 F1-Score) over balanced metrics. |

**Act on Insights**:
- If interpretability matters, stick with our XGBoost or Random Forest over deep learning models.
- Experiment with feature selection inspired by Hennebelle et al. to potentially boost accuracy.

## Take It Further
Improve the models with these next steps:
- **Try Stacking**: Combine models using stacking to push accuracy higher.
- **Focus on Recall**: Implement cost-sensitive learning to prioritize diabetic case detection.
- **Tune Thresholds**: Adjust classification thresholds in the notebook to optimize F1-scores.
- **Add Features**: Create interaction terms (e.g., BMI * Age) and test their impact.
- **Test Generalizability**: Apply the models to other diabetes datasets.
- **Go Dynamic**: Set up real-time data pipelines for adaptive predictions.

## License
Licensed under the MIT License. See [LICENSE](LICENSE) for details.

## References
- **Dataset**: Smith, J.W., et al. (1988). "Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus." *Proceedings of the Symposium on Computer Applications in Medical Care*, 261-265.
- **Libraries**:
  - McKinney, W. (2010). "Data Structures for Statistical Computing in Python." *Proceedings of the 9th Python in Science Conference*, 51-56.
  - Harris, C.R., et al. (2020). "Array Programming with NumPy." *Nature*, 585, 357–362.
  - Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825-2830.
  - Hunter, J.D. (2007). "Matplotlib: A 2D Graphics Environment." *Computing in Science & Engineering*, 9(3), 90-95.
  - Waskom, M. (2021). "Seaborn: Statistical Data Visualization." *Journal of Open Source Software*, 6(60), 3021.
  - Lundberg, S.M., & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems*, 30.
  - Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

## Need Help?
Reach out:
- Email: [ghanbarpourshiadeh.amirali@gmail.com]

