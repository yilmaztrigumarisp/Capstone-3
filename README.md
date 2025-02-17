California House Price Analysis

**Background**

House prices in California vary significantly due to factors like location, income levels, and property characteristics. Understanding key price drivers can help improve pricing strategies and investment decisions.

**Goals:**

1. Analyze key factors influencing house prices
2. Compare rule-based (OLS) vs. machine learning models
3. Identify the best predictive model

* Rule-Based Approach (OLS)

A baseline Ordinary Least Squares (OLS) regression was used, achieving:

Train MAE: $51.5K | Test MAE: $50.4K
Train MAPE: 28.68% | Test MAPE: 28.39%

* Best Model: Optimized CatBoost

After extensive feature engineering & hyperparameter tuning, the final CatBoost model significantly outperformed OLS:

Test MAE: ~$35.2K
Test MAPE: ~18.75%
Test RMSE: ~54.2K
Training Time: ~1,522 sec

**Conclusion & Recommendations**

* Median income is the strongest predictor – higher income areas have much higher house prices.
* Ocean proximity impacts price – homes near the ocean tend to be valued significantly higher.
* Room density matters – high bedrooms per room ratios often correlate with lower house prices.

**Recommendation**

* Model & Improvement

1. Improve housing detail, geographical data, and density planning to increase accuracy.
2. Use CatBoost for price estimation in real estate applications.

* Business and Market Recommendations

1. For real estate investors: Use the model to identify undervalued areas with strong price growth potential.
2. For policymakers: Leverage insights from the model to assess housing affordability trends and plan urban development.

Try the Streamlit App: 👉 [Insert Your Link Here]
