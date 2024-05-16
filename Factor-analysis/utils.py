from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def load_file(uploaded_file):
    # try:
    #     df = pd.read_csv(uploaded_file, index_col=0)
    # except
    return pd.read_csv(uploaded_file,index_col=0)

def sidebar():

    if "file" not in st.session_state:
        # control on/off of displaying elements after the title section
        st.session_state["file"] = False

    with st.sidebar:
        st.title("Factor Analysis")
        uploaded_file = st.file_uploader(label="",type='csv')

        st.divider()
        st.write("Test with a sample data")
        with st.popover("Sample Data"):
            st.markdown("""
                        <div>
                            <h4>Sample dataset</h4>
                                <div class=description>
                                    <p>
                                        Sample dataset for analysis is obtained from
                                        <a href=https://vincentarelbundock.github.io/Rdatasets/datasets.html>Link</a>
                                    </p>
                                    <div>
                                        The dataset consists of 25 personality items representing 5 factors.<br>
                                        Response of each item is collected using a 6-point scale:<br>
                                        <div class=description>
                                            <ol style="font-weight: normal">
                                                <li>Very Inaccurate</li>
                                                <li>Moderately Inaccurate</li>
                                                <li>Slightly Inaccurate</li>
                                                <li>Slightly Accurate</li>
                                                <li>Moderately Accurate</li>
                                                <li>Very Accurate</li>
                                            </ol>
                                        </div>
                                    </div>
                                </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                        )

            st.write("Click the button to load a sample dataset")
            if st.button("Sample data"):
                uploaded_file = "./data/bfi.csv"

        if uploaded_file is not None:
            st.session_state["file"] = True
            return load_file(uploaded_file)

        else:
            pass



def check_nulls(df):
    n_nulls = df.isnull().sum().sum()

    if n_nulls != 0:
        return True
    else:
        return False

def subset_data(df, columns):
    return df.drop(columns, axis=1)


def preprocessing(df: pd.DataFrame, columns: list):

    st.markdown("")
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
    st.markdown("")

    st.markdown("**Check data table**")

    with st.expander("View DataFrame"):
        st.dataframe(df)

    st.markdown(f"<div class=description>DataFrame shape : <code>{df.shape}</code></div>",unsafe_allow_html=True)

    return df

def adequacy_test(df):

    st.subheader("A. Bartlett's test for Sphericity")
    chi_square_value,p_value=calculate_bartlett_sphericity(df.values)
    st.markdown(f"""
                    <div class=description>
                        Bartlett's test result: <br>
                        <ul>
                            <li>chi_square_value : &nbsp;{chi_square_value:.2f}, <br>
                            <li>p_value :&nbsp;&nbsp;&nbsp;&nbsp;{p_value:.4f}
                        </ul>
                    </div>
                """,
                unsafe_allow_html=True)
    st.subheader("B. Kaiser-Myer-Olkin test")
    kmo_all,kmo_model=calculate_kmo(df)
    st.markdown(f"""
                    <div class=description>
                        Kaiser-Myer-Olkin test result: <code>{kmo_model:.4f}</code>
                    </div>
                """,
                unsafe_allow_html=True)

    st.write("")

    if (p_value < 0.05) & (kmo_model > 0.6):
        st.success("The data passes the adequacy tests for factor anlysis", icon="✅")

    else:
        st.warning("Factor Analysis may not be appropriate for this dataset!")

def fit_factor_analyzer(df, n_factors: int,rotation=None):
    fa = FactorAnalyzer(n_factors=n_factors,rotation=rotation)
    fa.fit(df)

    return fa

def scree_plot(df, eigenvalue):
    fig = px.line(x=range(1,df.shape[1]+1),y=eigenvalue, markers=True)
    fig.update_layout(
        title  ={
            "text":"Scree Plot",
            "y":0.95,
            "x":0.5,
            "xanchor":"center",
            "yanchor":"top"
        },
        xaxis_title = "Factors",
        yaxis_title ="Eigenvalue"
    )
    fig.add_hline(y=1, line_dash="dash")

    return st.plotly_chart(fig)

def determine_n_factors(eigenvalue):
    return eigenvalue[eigenvalue > 1].size

def highlight_cells(val,min=0.5):

    color = "grey" if val > min else ''
    return "background-color: {}".format(color)

def extract_high_loadings_category(df, min: int =0.5):
    data = {}

    for col in df.columns:
        data[col] = df[df.loc[:,col] > min][col].index.tolist()

    return data

def high_loading_factors(df, min:int =0.5):

    data = extract_high_loadings_category(df, min)

    st.subheader("Features associated with Factor(s)")

    for key in data.keys():
        if data[key] != []:
            st.markdown(f"- {key} has High Loading Factor for {data[key]}",unsafe_allow_html=True)
        else:
            st.markdown(f"- {key} has no High Loading Factor",unsafe_allow_html=True)


def factor_analysis_summary(fa, columns):
    return pd.DataFrame(fa.get_factor_variance(), index=['SS Loadings','Proportion Variance','Cumulative Variance'],columns=columns)

def factor_loading_plot(df, X, Y):
    fig = px.scatter(data_frame=df, x=X, y=Y,text=df.index)
    fig.update_layout(
        title  ={
            "text":f"Factor Loading ({X} X {Y})",
            "y":0.95,
            "x":0.5,
            "xanchor":"center",
            "yanchor":"top"
        }
    )
    fig.update_traces(textposition='top center')

    return st.plotly_chart(fig)


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
