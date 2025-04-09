import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="Accesul la Internet", layout="wide")
st.title("🌐 Internetul în zilele de astăzi")
st.header("Cum ne influențează accesul la Internet?")

uploaded_file = st.file_uploader("Alege un fișier CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.replace("..", pd.NA, inplace=True)
    data = data.dropna(axis=1, how="all")

    st.session_state["df_internet"] = data
    st.success("✅ Fișierul a fost încărcat și salvat în sesiune pentru a fi folosit în celelalte pagini.")

    st.subheader("🔍 Primele rânduri din datele încărcate:")
    st.dataframe(data.head())

    st.subheader("🔎 Cum dorești să vizualizezi datele?")
    col1, col2 = st.columns(2)

    if "show_custom_columns" not in st.session_state:
        st.session_state.show_custom_columns = False
    with col1:
        if st.button("Afișează primele 3 coloane"):
            st.session_state.show_custom_columns = False
            st.subheader("Primele 3 coloane")
            st.dataframe(data.iloc[:, :3])
    with col2:
        if st.button("Alege câte coloane să vezi"):
            st.session_state.show_custom_columns = True
    if st.session_state.show_custom_columns:
        num_columns = st.number_input("Introdu numărul de coloane pe care vrei să le vezi:",
                                      min_value=1, max_value=data.shape[1], value=3)
        st.subheader(f"Primele {num_columns} coloane")
        st.dataframe(data.iloc[:, :num_columns])

    st.title("📊 Accesul la Internet în 2022 - Top 5 țări")
    data_grafic = data[['Country Name', 'Individuals using the Internet (% of population) 2022 ']]
    data_grafic = data_grafic.sort_values(by='Individuals using the Internet (% of population) 2022 ', ascending=False).head(5)
    data_grafic = data_grafic.set_index('Country Name')
    st.subheader("Topul țărilor după procentul populației cu acces la Internet:")
    st.bar_chart(data_grafic)

    st.write("📊 Datele pentru Top 5:")
    st.dataframe(data_grafic)

    fig5, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data_grafic['Individuals using the Internet (% of population) 2022 '], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title("Distribuția procentului populației care utilizează Internetul")
    ax.set_xlabel("Procentul populației (%)")
    ax.set_ylabel("Număr de țări")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig5)

    st.title("📱 Distribuția țărilor după numărul de abonamente mobile")
    date = data[['Country Name', 'Mobile cellular subscriptions (per 100 people) 2022 ']]
    date['Mobile cellular subscriptions (per 100 people) 2022 '] = pd.to_numeric(date['Mobile cellular subscriptions (per 100 people) 2022 '], errors='coerce')
    date = date.dropna()

    date_sub_50 = (date['Mobile cellular subscriptions (per 100 people) 2022 '] < 50).sum()
    date_peste_50 = (date['Mobile cellular subscriptions (per 100 people) 2022 '] > 50).sum()

    pie_data = pd.DataFrame({
        "Categorie": ['Sub 50 de abonamente', 'Peste 50 de abonamente'],
        "Numar tari": [date_sub_50, date_peste_50]
    })

    fig = px.pie(pie_data,
                 values='Numar tari',
                 names='Categorie',
                 title='Distribuția țărilor după numărul de abonamente mobile',
                 color_discrete_sequence=['#D4498C', '#A4D4B4'])
    st.plotly_chart(fig)

    tari_sub_50 = date[date['Mobile cellular subscriptions (per 100 people) 2022 '] < 50]
    st.write("📉 Țările cu mai puțin de 50 de abonamente:")
    st.dataframe(tari_sub_50)

    st.title("❗ Situație valori lipsă")
    valoriLipsa = data.isna()
    numarValoriLipsa = valoriLipsa.sum().sum()
    numarValoriExistente = data.size - numarValoriLipsa
    pie_data1 = pd.DataFrame({
        'Categorie': ['Valori lipsă', 'Valori existente'],
        'Număr valori': [numarValoriLipsa, numarValoriExistente]
    })
    fig1 = px.pie(pie_data1, values='Număr valori', names='Categorie', title='Valori lipsă vs. valori existente',
                  color_discrete_sequence=['#CA054D', '#8551B8'])
    st.plotly_chart(fig1)

    df_lipsa = pd.DataFrame({
        "Coloană": data.columns,
        "Număr Valori Lipsă": data.isna().sum().values
    })
    df_lipsa = df_lipsa[df_lipsa["Număr Valori Lipsă"] > 0]
    fig2 = px.bar(df_lipsa, x="Coloană", y="Număr Valori Lipsă",
                 title="Numărul de valori lipsă pentru fiecare coloană",
                 text_auto=True,
                 color_discrete_sequence=['#FF890A'])
    st.plotly_chart(fig2)

    st.title("🔢 Numărul de valori unice per coloană")
    valoriUnice = data.nunique()
    dfValoriUnice = pd.DataFrame({
        'Nume coloană': valoriUnice.index,
        'Număr valori unice': valoriUnice.values
    }).sort_values(by='Număr valori unice', ascending=False)
    fig3 = px.bar(dfValoriUnice, x='Număr valori unice', y='Nume coloană', orientation='h',
                  title='Numărul de valori unice în fiecare coloană', text_auto=True,
                  color_discrete_sequence=['#F1FEAF'])
    st.plotly_chart(fig3)

    st.title("🔍 Țări care încep cu litera 'I'")
    dfDateFiltrare = data[data['Country Name'].str.startswith('I', na=False)]
    dfDateFiltrare = dfDateFiltrare.iloc[1:11, [0, 2, 4]]
    st.dataframe(dfDateFiltrare)
    fig4 = px.scatter(
        dfDateFiltrare,
        x='Country Name',
        y=dfDateFiltrare.columns[2],
        title='Distribuție țări după numărul de abonamente mobile',
        labels={'Country Name': 'Țară', dfDateFiltrare.columns[2]: 'Număr abonamente'},
        color='Country Name'
    )
    st.plotly_chart(fig4)
else:
    st.info("📂 Încarcă fișierul pentru a începe analiza.")
