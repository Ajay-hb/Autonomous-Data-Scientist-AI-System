import shap


def generate_shap_values(model, X_train):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_train)

    return shap_values
