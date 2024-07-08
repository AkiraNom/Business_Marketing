import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
import uuid

st.set_page_config(layout="wide")

@st.cache_data
def load_file(file_path):
    try:
        df = pd.read_csv(file_path, index_col=0)
        return df
    except Exception as e:
        st.session_state["file"] = False
        st.error(f"Error loading file: {e}")
        return None

def sidebar():

    with st.sidebar:
        st.title("Conjoint Analysis")
        uploaded_file = st.file_uploader(label="#",type='csv')

        if uploaded_file is not None:
            st.session_state["file"] = True
            st.session_state["file_path"] = uploaded_file

        st.write("Click the button to load a sample dataset")
        if st.button("Sample data"):
            st.session_state["file"] = True
            st.session_state["file_path"] = "./data/sample_data.csv"

def check_nulls(df):
    n_nulls = df.isnull().sum().sum()

    if n_nulls != 0:
        return True
    else:
        return False

def subset_data(df, columns):
    return df.drop(columns, axis=1)


def preprocessing(df: pd.DataFrame, columns: list):

    st.markdown("#####")
    st.subheader("Preprocessing data")

    if check_nulls(df):
        st.markdown("""
                    <div class=description>
                        <p>The null values are removed from the dataset....</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
        df = df.dropna()

    if columns==[]:
        st.markdown("<div class=description>All columns in the dataset are used in analysis</div>", unsafe_allow_html=True)

    if columns != []:
        df = subset_data(df, columns)
        st.markdown(f"""
                        <div class=description>
                            Feature(s), <code>{columns}</code>, is/are removed from the dataset....
                        </div>
                    """,
                    unsafe_allow_html=True)
    st.markdown("####")

    st.markdown("**Check data table**")

    with st.expander("View DataFrame"):
        st.dataframe(df)

    st.markdown(f"<div class=description>DataFrame shape : <code>{df.shape}</code></div>",unsafe_allow_html=True)

    return df

def compute_part_worth(df, X_cols: list,Y_col: str):
    X = df[X_cols]
    Y = df[Y_col]
    lr = sm.OLS(Y,X).fit()

    st.session_state["model"] = lr

    with st.expander("View regression coefficient summary"):
        st.text(lr.summary())

    coef = lr.params.to_dict()

    return coef

def set_marker_color(df):

    colors = ["#636EFA" if x > 0 else "crimson" for x in df['part worth utility']]
    return colors


def plot_part_worth_utility(df):

    fig = px.bar(df,
                 orientation="h",
                 text_auto=".2f",
                 title="Part Worth Utility"
                )

    marker_color = set_marker_color(df)
    fig.update_traces(marker_color=marker_color, textfont_size=13, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="", yaxis_title="",showlegend=False)

    return fig

def add_row():
    element_id = uuid.uuid4()
    st.session_state["rows"].append(str(element_id))

def remove_attribute(row_id):
    st.session_state["rows"].remove(str(row_id))

def generate_attribute(row_id,df,coef):
    row_container = st.empty()
    row_columns = row_container.columns((2, 3, 1))
    with row_columns[0]:
        attribute_name = st.text_input("Attribute Name", key=f"txt_{row_id}")
    with row_columns[1]:
        attribute_levels = st.multiselect(
            "Choose attribute levels",
            df.columns.tolist(),
            default=None, key=f"nbr_{row_id}")
    with row_columns[2]:
        st.write("######")
        st.button("🗑️", key=f"del_{row_id}", on_click=remove_attribute, args=[row_id])
    return {"name": attribute_name,
            "levels": attribute_levels,
            "coef": [coef[level] for level in attribute_levels]
            }

def plot_relative_importance(df):
    fig = px.bar(df,
                 x= "Attribute Name",
                 y = "Relative Importance",
                 color="Attribute Name",
                 text_auto=".2f",
                 title="Relative Importance of Attributes"
                )

    fig.update_traces(textfont_size=13, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="", yaxis_title="",showlegend=False)
    return fig

def check_duplicated_level(df, attribute_levels):

    attribute_dict = {level: name for name, levels in zip(df['Attribute Name'], df['Attribute Levels']) for level in levels}
    attribute_matches = [attribute_dict[level] for level in attribute_levels if level in attribute_dict]
    duplicates = [attribute for attribute in set(attribute_matches) if attribute_matches.count(attribute) > 1]

    if duplicates:
        st.warning(f"""**WARNING**: The attribute levels belonging to the same attribute {duplicates} are selected""",  icon="⚠️")

def select_attribute_levels(df1,df2):

    selected_attribute_levels = st.multiselect("Select attribute levels", df1.index.tolist())
    attribute_levels = df1.index.tolist()

    option = [1 if x in selected_attribute_levels else 0 for x in attribute_levels]

    check_duplicated_level(df2,selected_attribute_levels)

    return selected_attribute_levels, option

def predict_total_utility_score(df1,df2):

    _attribute_levels, option = select_attribute_levels(df1,df2)

    model = st.session_state["model"]
    return model.predict(option)[0], _attribute_levels

def market_share_simulation(df, price_levels):
    df_logit = df.copy()
    for level in price_levels:
        df_logit[level] = 0

    dependent_variable = st.session_state["Y"]
    df_logit = df_logit.drop([dependent_variable],axis=1)
    X = df_logit
    model = st.session_state["model"]
    predictUtil = model.predict(X)
    df_logit.loc[:,"predicted_Utility"] = predictUtil

    utility_values = list(predictUtil)
    total_utility = 0
    for val in utility_values:
        total_utility += np.exp(val)
    market_shares = []
    for val in utility_values:
        probability = np.exp(val)/total_utility
        market_shares.append(probability*100)

    df_logit.loc[:,"market_share"] = market_shares
    df_logit = df_logit.drop(price_levels, axis=1)
    df_logit = df_logit.drop_duplicates()

    df_logit.loc[:,"product name"] = [f"Product_{i+1}" for i in range(len(df_logit))]

    return df_logit

def generate_level_selectbox(df:pd.DataFrame,attributes:dict, n_rows:int):
    row_names = [f'rows_{j}' for j in range(n_rows)]
    _rows = []
    user_choice = []
    for row_name in row_names:
        row_name = st.columns(4)
        _rows.append(row_name)
    _rows = [_r for _row in _rows for _r in _row]
    for _r, (attr, level) in zip(_rows, attributes.items()):
        user_choice.append(_r.selectbox(f"Select a level of attribute: {attr}",level))

    if all(user_choice):

        query = ' & '.join([f"{value} == 1" for value in user_choice])
        filtered_df = df.query(query)

        if not filtered_df.empty:
            cols = st.columns(3)
            with cols[1]:
                container = st.container(border=True)
                with container:
                    st.markdown(f"""<p style ='font-size:1.2em;font-weight:bold;text-align:center;'>
                                Prodcut Name : {filtered_df["product name"].values[0]}%</p>""",
                                unsafe_allow_html=True)

                    st.markdown(f"""<p style ='font-size:1.2em;font-weight:bold;text-align:center;'>
                                Market Share : {filtered_df["market_share"].values[0]:.4f}%</p>""",
                                unsafe_allow_html=True)

        else:
            st.write("No data found for the selected combination")

def extract_attribute_level_by_id(df):

    remove_cols_list = ['market_share','predicted_Utility']
    keep_cols = [item for item in df.columns.to_list() if item not in remove_cols_list]

    df.loc[:,"product name"] = [f"Product_{i+1}" for i in range(len(df))]
    df = df.sort_values('market_share', ascending=True)

    product_id = st.selectbox('Select a product ID:', sorted(df["product name"].tolist()))
    levels = [col for col in keep_cols if not df[(df["product name"]==product_id)&(df[col] ==1)].empty]

    # st.dataframe(df[df["product name"]==product_id])
    st.markdown(f"Attribute Levels with {product_id}:")
    st.code(levels)
    container = st.container(border=True)
    container.markdown(f"""<p style ='font-size:1.2em;font-weight:bold;'>Market Share :
            {df[df["product name"]==product_id]["market_share"].values[0]:.4f}%</p>""",
            unsafe_allow_html=True)

def plot_market_share(df):

    fig = px.bar(df.sort_values("market_share",ascending=True),
                 y="market_share",
                 x = "product name",
                 text_auto=".2f",
                 title='Estimated Market Share')
    fig.update_traces(textfont_size=13, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Product Name", yaxis_title="Market Sahre (%)",showlegend=False)

    return fig

st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                padding-left:40px;
            }
            ol li::marker {
                font-weight:bolder;
            }
            ol {
                padding-left: 10px;
                padding-bottom: 10px;
            }
            li {
                font-weight: normal;
            }
            .description {
                padding-left: 40px;
                padding-bottom: 10px;
            }
            </style>
            ''',
            unsafe_allow_html=True)
