import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Random Forest & XGBoost", layout="wide")
st.title("Modele de regresie: Random Forest & XGBoost")

st.markdown("""
În această secțiune aplicăm două modele de regresie — **Random Forest** și **XGBoost** — pentru a analiza și prezice valoarea variabilei dependente.

📌 **Variabila dependentă** (sau *target*) este:
**`Individuals using the Internet (% of population) 2022 `**

Aceasta reflectă procentul populației care utilizează Internetul într-o țară în anul 2022 și reprezintă **fenomenul pe care dorim să-l explicăm** în funcție de alți factori (precum numărul de servere securizate, abonamente mobile, etc.).

- **Random Forest**: un model bazat pe mai mulți arbori de decizie, ideal pentru a surprinde relații non-lineare între variabile.
- **XGBoost**: un model performant de boosting care construiește arbori secvențial, optimizând erorile anterioare.
""")

if "df_internet" not in st.session_state:
    st.warning("⚠️ Te rugăm să încarci fișierul CSV din pagina principală.")
    st.stop()

df = st.session_state["df_internet"].copy()
df.replace("..", pd.NA, inplace=True)
df = df.dropna(axis=1, how="all")

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

def_target = 'Individuals using the Internet (% of population) 2022 '
target = st.selectbox("🎯 Alege coloana țintă (target):", options=numeric_cols, index=numeric_cols.index(def_target) if def_target in numeric_cols else 0)

if target:
    df_model = df[numeric_cols + cat_cols].dropna(subset=[target])
    df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]
    X = X.select_dtypes(include=[np.number])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("Antrenare modele de regresie...")

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    def eval_model(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        st.markdown(f"### 📌 {name}")
        st.write(f"• Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"• Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"• Coeficient R²: {r2:.4f}")
        st.markdown("---")
        return pd.DataFrame({"Valori reale": y_true, "Predicții": y_pred})

    st.subheader("Rezultatele modelelor:")
    df_rf = eval_model("Random Forest", y_test, rf_pred)
    df_xgb = eval_model("XGBoost", y_test, xgb_pred)

    st.subheader("📈 Comparație vizuală: Predicții vs. Valori Reale (XGBoost)")
    st.line_chart(df_xgb.reset_index(drop=True))
    st.caption("🔍 Graficul compară valorile reale cu cele prezise de modelul XGBoost pentru a evidenția cât de bine s-a potrivit modelul cu datele reale.")

    st.subheader("🔍 Importanța caracteristicilor (Random Forest)")
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(importances)
    st.caption("📊 Această diagramă arată ce variabile au avut cea mai mare influență în predicția procentului populației care utilizează Internetul.")

    st.subheader("📌 Concluzii")
    st.markdown("""
    - Am antrenat două modele robuste pentru regresie, iar scorurile R² oferă o idee despre cât de bine sunt explicate variațiile în date.
    - Un scor R² apropiat de 1 indică o potrivire bună; valori negative semnalează un model ineficient.
    - Din graficul de importanță vedem ce factori (ex. abonamente mobile, servere securizate) influențează cel mai mult utilizarea Internetului.
    - Modelul poate fi folosit pentru a înțelege care sunt prioritățile pentru a îmbunătăți accesul la Internet în diverse țări.
    """)
