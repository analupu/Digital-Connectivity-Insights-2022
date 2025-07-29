import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Regresie Liniară", layout="wide")
st.title("Model de Regresie Liniară cu Encoding")

st.markdown("""
În această pagină, construim un **model de regresie liniară** pentru a prezice procentul populației care utilizează Internetul în 2022,
în funcție de alți indicatori tehnologici. Un aspect important aici este **encoding-ul** variabilelor categorice.

**Encoding** înseamnă transformarea coloanelor de tip text (cum ar fi `Country Name`, `Country Code`) în coloane numerice prin crearea de **variabile binare (dummy variables)**,
pentru ca modelul să poată înțelege aceste informații.

Modelul liniar presupune o relație liniară între variabila dependentă și predictorii săi.
""")

st.markdown("""
💡 **Știați că...?**

Modelele de regresie liniară pot procesa doar **valori numerice**. Prin urmare, orice coloană cu text (precum numele țării) trebuie convertită înainte.
Encoding-ul transformă aceste valori în coloane binare pentru fiecare categorie — de exemplu, o coloană `Country Name_Austria` care ia valoarea 1 dacă țara este Austria, altfel 0.

Această tehnică este esențială deoarece evită atribuirea unor relații numerice arbitrare între categorii. De exemplu, nu vrem ca modelul să interpreteze că „Austria = 1” și „Brazilia = 2” înseamnă că Brazilia este „mai mare” decât Austria.
""")

if "df_internet" not in st.session_state:
    uploaded_file = st.file_uploader("Încarcă fișierul CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df_internet = df
    else:
        st.warning("⚠️ Te rugăm să încarci fișierul CSV pentru a continua.")
        st.stop()
else:
    df = st.session_state.df_internet.copy()

df.replace("..", pd.NA, inplace=True)
df = df.dropna(axis=1, how="all")

numeric_cols = [
    'Secure Internet servers (per 1 million people) 2022',
    'Individuals using the Internet (% of population) 2022 ',
    'Mobile cellular subscriptions (per 100 people) 2022 '
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

model_df = df[['Country Name', 'Country Code'] + numeric_cols].dropna()
model_df = model_df.reset_index(drop=True)

st.subheader("🔍 Datele folosite pentru model:")
nr_randuri = st.slider("Selectează câte rânduri vrei să vizualizezi:",
                       min_value=5,
                       max_value=len(model_df),
                       value=10,
                       step=5)

st.dataframe(model_df.iloc[:nr_randuri])

st.markdown(""" **Encoding aplicat:**
Am transformat coloanele `Country Name` și `Country Code` folosind `pd.get_dummies` cu `drop_first=True` pentru a evita colinearitatea.
""")

df_encoded = pd.get_dummies(model_df, columns=['Country Name', 'Country Code'], drop_first=True)

target = 'Individuals using the Internet (% of population) 2022 '
X = df_encoded.drop([target], axis=1)
y = df_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
st.session_state.lr_predictions = pd.DataFrame({
    "Valori reale": y_test.values,
    "Predicții": y_pred
})

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Rezultatele modelului:")
st.write(f"**MAE (Mean Absolute Error):** {mae:.2f}")
st.write(f"**MSE (Mean Squared Error):** {mse:.2f}")
st.write(f"**R² Score:** {r2:.4f}")


st.subheader("📉 Comparație: valori reale vs. predicții")
st.line_chart(st.session_state.lr_predictions.reset_index(drop=True))

st.subheader("📌 Concluzii")
st.markdown("""
- Am construit un model de regresie liniară folosind date prelucrate numeric.
- Encoding-ul a fost esențial pentru a putea include țările în model fără a introduce relații numerice false.
- Scorul R² ne arată cât de bine explică modelul variația accesului la Internet. Un scor apropiat de 1 este ideal.
- Acest model oferă o perspectivă de bază asupra relațiilor liniare, dar pentru o performanță mai bună sunt recomandate modele mai complexe (Random Forest, XGBoost).
""")