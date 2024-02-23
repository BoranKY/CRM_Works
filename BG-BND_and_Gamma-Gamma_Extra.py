######################################## CLTV Estimation with BG-NBD and Gamma-Gamma #######################################
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
pd.set_option("display.float_format",lambda x: "%.4f" % x)

df_ = pd.read_excel("online_retail_II.xlsx",sheet_name="Year 2010-2011")
df = df_.copy()


####################################### Preparing the Data #######################################

def outlier_treshold(dataframe,variable):
    quartile1 = dataframe[ variable ].quantile(0.01)
    quartile3 = dataframe[ variable ].quantile(0.99)
    inter_quartile = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * inter_quartile
    low_limit = quartile1 - 1.5 * inter_quartile
    return up_limit,low_limit


def replace_with_treshold(dataframe,variable):
    up_limit,low_limit = outlier_treshold(dataframe,variable)
    df.loc[ (df[ variable ] > up_limit),variable ] = up_limit
    df.loc[ (df[ variable ] < low_limit),variable ] = low_limit


df.describe().T
df.isnull().sum()

df.dropna(inplace=True)
df = df[ ~df[ "Invoice" ].str.contains("C",na=False) ]
replace_with_treshold(df,"Quantity")
replace_with_treshold(df,"Price")
df = df[ df[ "Price" ] > 0 ]
df[ "Monetray" ] = df[ "Quantity" ] * df[ "Price" ]

df[ "InvoiceDate" ].max()
today = dt.datetime(2011,12,11)

cltv = df.groupby("Customer ID").agg({"InvoiceDate": [ lambda x: (x.max() - x.min()).days,
                                                       lambda x: (today - x.min()).days ],
                                      "Invoice": lambda y: y.nunique(),
                                      "Monetray": lambda z: z.sum()})

cltv.columns = cltv.columns.droplevel(0)

cltv.columns = ["recency","T","frequency","monetary"]
cltv= cltv[cltv["recency"] > 0]
cltv= cltv[cltv["frequency"] > 1]

cltv["recency"] = cltv["recency"] / 7
cltv["T"] = cltv["T"] / 7
cltv["monetary"] = cltv["monetary"] / cltv["frequency"]

########################### CREATING BGF MODEL ###########################
bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(cltv["frequency"],cltv["recency"],cltv["T"])


cltv["expectec_1_month"] = bgf.predict(4,cltv["frequency"],cltv["recency"],cltv["T"])
cltv["expected_12_month"] = bgf.predict(4*12,cltv["frequency"],cltv["recency"],cltv["T"])

plot_period_transactions(bgf)
plt.show()

########################### CREATING GAMMA GAMMA SUBMODEL  ###########################
ggm = GammaGammaFitter(penalizer_coef=0.001)
ggm.fit(cltv["frequency"],cltv["monetary"])

cltv["expected_average_profit"] = ggm.conditional_expected_average_profit(cltv["frequency"],cltv["monetary"])

########################### CREATING CLTV FOR 6 MONTH   ###########################
cltv_six_mont = ggm.customer_lifetime_value(bgf,
                            cltv["frequency"],
                            cltv["recency"],
                            cltv["T"],
                            cltv["monetary"],
                            freq="W",
                            time=6,
                            discount_rate=0.01)


cltv_six_mont.reset_index()
cltv_final = cltv.merge(cltv_six_mont, on="Customer ID", how="left")
########################### CREATING SEGMENTS OF CLVT_6_MONTH  ###########################
cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D","C","B","A"])