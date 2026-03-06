import pandas as pd

def generate_retention_recommendations(model, X_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    results = pd.DataFrame({
        "Churn_Probability": probabilities
    })

    def classify_risk(prob):
        if prob >= 0.75:
            return "High Risk"
        elif prob >= 0.50:
            return "Medium Risk"
        else:
            return "Low Risk"

    results["Risk_Level"] = results["Churn_Probability"].apply(classify_risk)

    def retention_action(risk):
        if risk == "High Risk":
            return "Immediate 30% Discount + Personal Call"
        elif risk == "Medium Risk":
            return "Send Personalized Offer Email"
        else:
            return "Provide Loyalty Points Reward"

    results["Recommended_Action"] = results["Risk_Level"].apply(retention_action)

    print("\n🔥 Retention Strategy Sample:\n")
    print(results.head())

    return results
