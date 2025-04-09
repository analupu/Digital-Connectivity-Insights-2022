import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import plotly.express as px
import seaborn as sb

st.set_page_config(page_title="Grafice", layout="wide")
st.title("Grafice și Analize Vizuale")

if "df_internet" not in st.session_state:
    st.warning("⚠️ Te rugăm să încarci fișierul CSV din pagina principală.")
    st.stop()
data = st.session_state["df_internet"].copy()
data.replace("..", pd.NA, inplace=True)
data = data.dropna(axis=1, how="all")

df_numeric = data.copy()
for col in df_numeric.columns:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

numerical_cols = df_numeric.select_dtypes(include='number').dropna(axis=1, how='all').columns.tolist()

if numerical_cols:
    n_cols = 3
    n_rows = math.ceil(len(numerical_cols) / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axs = axs.flatten()

    for i, col in enumerate(numerical_cols):
        axs[i].hist(df_numeric[col].dropna(), bins=30, edgecolor='black', color='skyblue')
        axs[i].set_title(f'Distribuția: {col}')
        axs[i].set_xlabel(col)
        axs[i].set_ylabel('Frecvență')

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    st.subheader("📊 Histograme pentru variabilele numerice")
    st.pyplot(fig)
else:
    st.warning("Nu există coloane numerice disponibile pentru histograme.")

st.subheader("🗺️ Hartă - Acces la Internet")
data['Individuals using the Internet (% of population) 2022 '] = pd.to_numeric(
    data['Individuals using the Internet (% of population) 2022 '], errors='coerce')

fig = px.choropleth(
    data_frame=data,
    locations='Country Name',
    locationmode='country names',
    color='Individuals using the Internet (% of population) 2022 ',
    hover_name='Country Name',
    color_continuous_scale='Blues',
    title='Procentul populației care folosește Internetul în 2022'
)
st.plotly_chart(fig)
st.caption("🔵 În această hartă interactivă sunt colorate țările în funcție de procentul populației care folosește Internetul în 2022. Cu cât nuanța de albastru este mai intensă, cu atât procentul este mai mare.")

def find_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_df = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return lower_bound, upper_bound, outliers_df

col = 'Mobile cellular subscriptions (per 100 people) 2022 '
data[col] = pd.to_numeric(data[col], errors='coerce')

if col in data.columns:
    lower, upper, outliers_df = find_outliers_iqr(data, col)

    st.subheader(f"📈 Outlieri pentru: {col}")
    st.markdown(f"- **Limita inferioară:** {lower:.2f}")
    st.markdown(f"- **Limita superioară:** {upper:.2f}")
    st.markdown(f"- **Număr de outlieri:** {len(outliers_df)}")

    if not outliers_df.empty:
        st.write("🔍 Tabel cu outlieri:")
        st.dataframe(outliers_df[['Country Name', col]])
        st.caption("📌 Tabelul de mai sus conține țările care au un număr de abonamente mobile considerat ieșit din comun (mult mai mic sau mai mare decât restul).")
    else:
        st.info("Nu s-au identificat outlieri în această coloană.")

    fig, ax = plt.subplots(figsize=(8, 4))
    sb.boxplot(x=data[col], ax=ax, color='lightblue')
    ax.set_title(f"Boxplot pentru: {col}")
    st.pyplot(fig)
    st.caption("📦 Boxplot-ul evidențiază distribuția valorilor și valorile extreme (outlieri) sub forma cercurilor aflate în afara barelor verticale.")

st.title('🔗 Corelații între variabile')
selected_cols = st.multiselect("Selectează variabilele pentru analiza de corelație:", options=numerical_cols,
                               default=numerical_cols)
if len(selected_cols) >= 2:
    corr_matrix = df_numeric[selected_cols].corr()

    st.write("### 🔢 Tabel Corelație:")
    st.dataframe(corr_matrix)

    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    ax_corr.set_title("Matricea de corelație pentru variabilele selectate")
    st.pyplot(fig_corr)
    st.caption("🎯 Matricea de corelație arată intensitatea relațiilor dintre variabile. Valori apropiate de 1 sau -1 indică o corelație puternică pozitivă sau negativă.")
else:
    st.warning("Selectează cel puțin 2 coloane pentru analiza de corelație.")
