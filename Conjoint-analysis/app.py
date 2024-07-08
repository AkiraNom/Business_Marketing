import pandas as pd
import streamlit as st

from utils import (
    sidebar,
    load_file,
    check_nulls,
    compute_part_worth,
    plot_part_worth_utility,
    add_row,
    generate_attribute,
    plot_relative_importance,
    predict_total_utility_score,
    market_share_simulation,
    select_attribute_levels,
    generate_level_selectbox,
    extract_attribute_level_by_id,
    plot_market_share
    )

# initialize session_state
if "file_path" not in st.session_state:
    st.session_state["file_path"] = None

if "file" not in st.session_state:
    # control on/off of displaying elements after the title section
    st.session_state["file"] = False

## Side bar

sidebar()

# test purpose: ###############
st.session_state["file"] = True
st.session_state["file_path"] = "./data/sample_data.csv"
###################

## Main window

st.title('Conjoint Analysis')

st.markdown("""Conjoint analysis is a common statistical method of pricing and product research.
            The method uncovers customers' choices through market surverys.""")

if st.session_state["file"]:
    pass

else:
    st.write("##")
    st.warning("Please select a file to analyze from the sidemenu", icon="⚠️")
    st.stop()

###1. Data Table###
st.header('1. Data Table')

df = load_file(st.session_state["file_path"])

with st.expander("View DataFrame"):
    st.dataframe(df)

st.markdown(f"<div class=description>DataFrame shape : <code>{df.shape}</code></div>", unsafe_allow_html=True)


###2. Preprocessing Data###
st.header("2. Preprocessing Data",divider="grey")
st.markdown("""
            <div class=description>
                The preprocessing step performs:
                <div>
                    <ol>
                        <li> Remove null values </li>
                        <li> Define dependent and independent variables</li>
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
st.write("####")

st.subheader("Define dependent/independent variables for analysis")

if "X" not in st.session_state:
    st.session_state["X"] = None
if "Y" not in st.session_state:
    st.session_state["Y"] = None
# set the last col as a default dependent variable (Y)
if "selectbox_option" not in st.session_state:
    st.session_state["selectbox_option"] = len(df.columns.tolist())-1
if "model" not in st.session_state:
    st.session_state["model"] = None

cols = st.columns([1,0.3])
with cols[1]:
    st.write("######")
    select_all = st.checkbox('select all')
    if select_all:
        st.session_state['X'] = df.columns.tolist()
    else:
        # st.session_state['X'] = None
        # for testing############
        st.session_state["X"] = df.columns.tolist()[:-1]
        #####################
with cols[0]:
    X_cols = st.multiselect(
        "Choose independent variables (X)",
        df.columns.tolist(),
        default=st.session_state['X'])
Y_col = st.selectbox(
    "Choose a dependent variable (Y)",
    options=df.columns.tolist(),
    index=st.session_state['selectbox_option']
    )

if Y_col in X_cols:
    st.error("ERROR: Selected dependent variable is included in the independet variables list", icon="🚨")
    st.stop()

else:
    st.session_state.X = X_cols
    st.session_state.Y = Y_col

if X_cols:
    # 3. compute part worth utility score
    st.header("3. Compute Part Wortth Utility Score")
    coef = compute_part_worth(df,X_cols,Y_col)
else:
    st.stop()

cols = st.columns([1,3])
with cols[0]:
    df_coef = pd.DataFrame(coef, index=['part worth utility']).T
    st.dataframe(df_coef,use_container_width=True)
with cols[1]:
    st.plotly_chart(plot_part_worth_utility(df_coef.iloc[::-1]))

# 4. Define attribute and attribute levels
st.header("4. Define attribute and attribute levels")
if "rows" not in st.session_state:
    st.session_state["rows"] = []

rows_collection = []

for row in st.session_state["rows"]:
    row_data = generate_attribute(row, df, coef)
    rows_collection.append(row_data)

menu = st.columns([1,2])

with menu[0]:
    st.button("Add Attribute", on_click=add_row)
    if len(rows_collection) > 0:
        st.markdown("**Attributes**")
        data = pd.DataFrame(rows_collection)
        data.rename(columns={"name": "Attribute Name", "levels": "Attribute Levels", "coef":"Attribute Coefficient"}, inplace=True)
        data["Part Worth Range"] = data["Attribute Coefficient"].apply(lambda x: max(x) - min(x) if x else None )
        if data['Attribute Coefficient'] is not None:
            total_range = data["Part Worth Range"].sum()
            data["Relative Importance"] = data["Part Worth Range"]/total_range*100

            st.dataframe(data=data[['Attribute Name','Part Worth Range','Relative Importance']],use_container_width=True,hide_index=True)

    else:
        st.error("Please define attributues and their attribute levels", icon="🚨")
        st.stop()

    with menu[1]:
        st.write("######")
        st.plotly_chart(plot_relative_importance(data))

with st.expander("View data table"):
    st.dataframe(data=data, use_container_width=True, hide_index=True)

if data['Attribute Name'].to_list():
    st.subheader('Dollar cost per utility score ')
    cols = st.columns([0.2,1,1,1.5])
    with cols[1]:
        cost_col = st.selectbox("Select a price column", data['Attribute Name'].to_list())
        st.write("")
        cost_difference = st.number_input("Type dollar cost",step=5.0,help="The difference between max and min price in analysis")
    with cols[2]:
        st.write("")
        unit_price = cost_difference/float(data[data["Attribute Name"] == cost_col]["Part Worth Range"].iloc[0])
        container = st.container()
        container.markdown(f"""<div style='font-size:1.1em;text-align:center;'> <b>Dollar cost per unit utility score:</b></div>""",unsafe_allow_html=True)
        container.markdown(f"""<div style='font-size:2em;text-align:center;text-decoration: underline;'><b>${unit_price:.2f}</b></div>""", unsafe_allow_html=True)
        st.write("#####")
        with st.popover('Methoddollar cost per unit utility score'):
            st.markdown(r"""
                        1. Find the difference betwwen max and min price
                        2. Compute the attribute utility score of price

                            $UtilityCost = \frac{DifferencePrice}{AttributeUtilityScore}$

                        """,
                        unsafe_allow_html=True
                        )

# 5. predict total utility socres
st.header("5. Predict total utility scores and optimal price point")

cols = st.columns([0.2,2.5,1,1])
with cols[1]:
    predicted_total_utility, selected_level = predict_total_utility_score(df_coef,data)
    optimal_price = unit_price * predicted_total_utility

    container = st.container(border=True)
    with container:
        cols = st.columns([0.2,1,0.1,1,0.2])
        with cols[1]:
            st.markdown(f"""<div style='font-size:1.2em;text-align:center;'> <b>Predictd total utility score:</b></div>""",unsafe_allow_html=True)
            st.markdown(f"""<div style='font-size:3em;color:#5c6ac4;text-align:center;'><b>{predicted_total_utility:.3f}</b></div>""", unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f"""<div style='font-size:1.2em;text-align:center;'> <b>Optimal price point:</b></div>""",unsafe_allow_html=True)
            st.markdown(f"""<div style='font-size:3em;color:#5c6ac4;text-align:center;'><b>${optimal_price:.2f}</b></div>""", unsafe_allow_html=True)

# 6. estimate market share
st.header("6. Market Share Simulation using Logit-choice rule")
with st.popover("Details in market share simulation"):
    st.markdown(r"""
        Calculate the probability that the customer will choose that a particular feature-bundle.

        In order to simulate the design patter, we have to specify a choice rule to transform part-worth utility into the produc choice that customers are most likely to make. <br>
        The three most commont rules are:
        1. Maximum utility

        2. Share of utility

        3. logit

        The maximum utility rule is simple, but it tends to predict more extreme market shares. i.e. closer to 0% or 100% market shares than other rules. This rule is also less robust. Share of uitlity and logit rules are sensitive to the scale range on which utility is measured.

        **Logit-choice rule**

        Assumption: the utilities follow a random process

        This rule states that the probability of a customer choosing a particular feature bundle is proportional to the exponential of the uitlity of that the bundle.

        >$P_j = \frac{e^{U_j}}{\sum^n_{k=1}{e^{U_k}}}$

        Where $U_j$ is the utility of bundle $j$, and the denominator is the sum of the exponentials of the utilities of all bundles being considered.

        This approach accounts for the fact that customers are more likely to choose bundles with higher utility values, and it provides a way to estimate the maket share foe each possible bundle.

        Reference:

        “[Conjoint Analysis: Marketing Engineering Technical Note](https://faculty.washington.edu/sundar/NPM/CONJOINT-ProductDesign/TN09%20-%20Conjoint%20Analysis%20Technical%20Note.pdf)” supplement to Chapter 6 of Principles of Marketing Engineering, by Gary L. Lilien, Arvind Rangaswamy, and Arnaud De Bruyn (2007).

                """, unsafe_allow_html=True)

price_levels = data[data["Attribute Name"]==cost_col]["Attribute Levels"].iloc[0]
df_logit = market_share_simulation(df, price_levels)

cols = st.columns([3,1])
with cols[0]:
    st.plotly_chart(plot_market_share(df_logit))
with cols[1]:
    st.write("#####")
    st.markdown("<b>Search by Product ID </b>", unsafe_allow_html=True)
    extract_attribute_level_by_id(df_logit)

with st.expander("View Data Table"):
    st.dataframe(df_logit)

attributes = data.set_index("Attribute Name").to_dict()["Attribute Levels"]
del attributes[cost_col]
n_attributes = len(attributes)

if (n_attributes > 5) & (n_attributes%4 != 0):
    n_rows = int(n_attributes/4 + 1)
elif n_attributes < 5:
    n_rows = 1
else:
    n_rows = int(n_attributes/4)

st.write("####")
st.markdown("<b>Search by Attribute Levels </b>", unsafe_allow_html=True)
generate_level_selectbox(df_logit,attributes,n_rows)


#7. misconception
st.header("7. Misconceptions about Conjoint Analysis")
st.markdown("""
        Three common mistakes:
        1. Conjoint will estimate a market share <br>

        Conjoint analysis is used to assess preferences, and preferences are highly correlated with sales. But they are not identical to sales. <br>

            > In short, I always talk about "relative share of preference" unless I have strong evidence and specific intent to assess market share.
        <br>

        2. Conjoint Gives simple pricing data <br>

        Motivation for conjoint analysis is to get insight on pricing. However, **conjoint does not directly answer, 'How much can we charge'** or **'How much is this feature worth?'**
        To use conjoint for successful pricing research, you need repeated observations to understand how prices work for your category, product, market, and brand. That requires iterative data, attention to the results, and rational modeling of effects.

        3. Highest Preference == Best Product <br>

        **Stakeholders believe that the best product decision is to offer the most-preferred feature(s)**. For instance, you get a highest score for black color of a car. Would producing a black car maximize a profit? It may not be true since this analysis assumes that there is no competition. What if all availble cars in the world are black?You won't have good profit from this product.

        Referrence:
        [Misconceptions about Conjoint Analysis](https://quantuxblog.com/misconceptions-about-conjoint-analysis)

        """,
        unsafe_allow_html=True)
