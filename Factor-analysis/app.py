import streamlit as st
import pandas as pd

from utils import (
    sidebar,
    load_file,
    check_nulls,
    preprocessing,
    adequacy_test,
    fit_factor_analyzer,
    scree_plot,
    determine_n_factors,
    highlight_cells,
    high_loading_factors,
    factor_analysis_summary,
    factor_loading_plot
)

if "file_path" not in st.session_state:
    st.session_state["file_path"] = None

if "file" not in st.session_state:
    # control on/off of displaying elements after the title section
    st.session_state["file"] = False

###Side Bar###

sidebar()

###Main Window###

st.title("Factor Analysis")
st.markdown("""Factor analysis is a sophisticated statistical method aimed at reducing a large number of variables into a smaller set of factors. This technique is valuable for extracting the maximum common variance from all variables, transforming them into a single score for further analysis.\n\n **Reference** : [Comprehensive Guide to Factor Analysis](https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/factor-analysis/)""")


if st.session_state["file"]:
    pass

else:
    st.write("##")
    st.warning("Please select a file to analyze from the sidemenu")
    st.stop()

###1. Data Table###
st.header('1. Data Table')

df=load_file(st.session_state["file_path"])

with st.expander("View DataFrame"):
    st.dataframe(df)

st.markdown(f"<div class=description>DataFrame shape : <code>{df.shape}</code></div>", unsafe_allow_html=True)


###2. Preprocessing Data###
st.header("2. Preprocessing Data",divider="grey")
st.write("")
st.markdown("""
            <div class=description>
                The preprocessing step performs:
                <div>
                    <ol>
                        <li> Remove null values </li>
                        <li>Drop unecessary columns</li>
                    </ol>
                </div>
            </div>""",
            unsafe_allow_html=True)

st.subheader("Check Missing Value Handling")
st.markdown(f"""
                <div class=description>
                    Does the dataset contain missing values? : <code>{check_nulls(df)}</code>
                </div>
            """,
            unsafe_allow_html=True)

st.subheader("Select columns to be removed from analysis", help='Leave this empty if all columns are used in analysis')


if "opitions" not in st.session_state:
    st.session_state["options"] = None

options = st.multiselect(
    "Select columns to be removed",
    df.columns.tolist(),
    default=st.session_state['options'])

st.session_state.options = options

if len(st.session_state.options) != 0:
    st.markdown(f"<div class=description>Selected column(s) is/are <code>{st.session_state.options}</code></div>", unsafe_allow_html=True)
else:
    pass

df=preprocessing(df, options)

###3. Adequacy Test###
st.header("3. Adequacy Test",divider="grey")
with st.popover("Explaination for Adequacy Test"):
    st.markdown("""
                To perform factor analysis, you need to evaluate the factorability of the data.There are two methods to evaluate the factorability.

                1.  Bartlett's Test for Sphericity
                2. Kaiser-Meyer-Olkin Test

                **Bartlett's test for sphericity**

                Bartlett's test of sphericity tests whether a matrix (of correlations) is significantly different from an identity matrix (filled with 0). It tests whether the correlation coefficients are all 0. The test computes the probability that the correlation matrix has significant correlations among at least some of the variables in a dataset, a prerequisite for factor analysis to work. If the test fails to reject the null hypothesis, you should not employ a factor analysis.Keep in mind that as the sample size increases, this test tends to be always significant, which makes it not particularly useful or informative in well-powered studies.

                **Kaiser-Meyer-Olkin test**

                The Kaiser-Meyer-Olkin (KMO) statistic, which can vary from 0 to 1, indicates the degree to which each variable in a set is predicted without error by the other variables. <br>
                A value of 0 indicates that the sum of partial correlations is large relative to the sum correlations, indicating factor analysis is likely to be inappropriate. A KMO value close to 1 indicates that the sum of partial correlations is not large relative to the sum of correlations and so factor analysis should yield distinct and reliable factors. Value less than 0.6 is considered inadequate.

                """,
                unsafe_allow_html=True)
st.write("")

adequacy_test(df)


###4. select factors###
help_description_adequacy_test = "If a dataset does not satisfy underlying assumptions to perform a factor analysis, obtained results may not be reliable. See more details [here](https://www.publichealth.columbia.edu/research/population-health-methods/exploratory-factor-analysis)"
if st.session_state["adequacy_test"] or st.checkbox("Continue performing a factor analysis?",help=help_description_adequacy_test):
    pass
else:
    st.stop()

n_factors_description = "select the number of factors to be equal to the number of eigenvalues greater than or equal to one[]"
st.header("4. Select the number of factors", divider='grey',help=n_factors_description)
fa = fit_factor_analyzer(df, n_factors=25)
ev, v = fa.get_eigenvalues()

scree_plot(df, ev)
st.info(n_factors_description)
n_factors_scree = determine_n_factors(ev)

cols = st.columns(2)
with cols[0]:
    st.markdown(f"""
                    <div class=description>
                        Proposed the number of factors from the scree plot&nbsp;: <code>{n_factors_scree}</code>
                        <br>
                    </div>
                """,
                unsafe_allow_html=True)
with cols[1]:
    if st.checkbox("Manually define the number"):
        n_factors = st.slider("Choose a number of factors", min_value=0, max_value=10, value=n_factors_scree)
    else:
        n_factors = n_factors_scree


###5. Factor Analysis###
st.header("5. Factor Analysis",divider="grey")
st.write("####")

fa = fit_factor_analyzer(df, n_factors=n_factors,rotation='varimax')

cols = [f'Factor{x}' for x in range(1,n_factors+1)]
df_factor = pd.DataFrame(data=fa.loadings_, index=df.columns, columns=cols)

with st.expander("View the DataFrame"):
    st.dataframe(df_factor.style.map(highlight_cells))

high_loading_factors(df_factor)

st.subheader("Summary Table")
st.dataframe(factor_analysis_summary(fa,cols))

st.write()

###6. Inspect Factors###
st.header("6. Inspect Factors", divider="grey")
st.write("####")
cols = st.columns(3)
with cols[0]:
    X = st.selectbox("Select X-value", df_factor.columns.tolist(), index=0)
with cols[2]:
    Y = st.selectbox("Select Y-value", df_factor.columns.tolist(), index=1)

factor_loading_plot(df_factor, X, Y)
