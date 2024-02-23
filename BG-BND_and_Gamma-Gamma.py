######################################## CLTV Estimation with BG-NBD and Gamma-Gamma #######################################
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
pd.set_option("display.float_format",lambda x:"%.4f" % x)
####################################### Preparing the Data #######################################
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()
df.head()

def outlier_tresholds(dataframe,value):
    quartile1 = dataframe[value].quantile(0.01)
    quartile3 = dataframe[value].quantile(0.99)
    inter_quartile = quartile3 - quartile1
    up_limit = (quartile3 + 1.5* inter_quartile).round()
    low_limit = (quartile1 - 1.5* inter_quartile).round()
    return  up_limit,low_limit

def replace_with_tresholds(dataframe,value):
    up_limit,low_limit = outlier_tresholds(dataframe,value)
    dataframe.loc[(dataframe[value] < low_limit), value] = low_limit
    dataframe.loc[ (dataframe[ value ] > up_limit),value ] = up_limit


df.describe().T

cat = ["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online"]
for x in cat:
    replace_with_tresholds(df,x)

df.head()
df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_spent"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.info()

change = df.columns[df.columns.str.contains("date")]

for x in change:
    df[ x ] = df[x].apply(pd.to_datetime)



####################################### Creating the CLTV Data Structure #######################################

df.head()
df["last_order_date"].max()
today = dt.datetime(2021,6,1)

cltv = pd.DataFrame()
cltv["customer_id"] = df["master_id"]
cltv["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).dt.days) /7
cltv["T_weekly"] = ((today - df["first_order_date"]).dt.days) /7
cltv["frequency"] = df["total_order"]
cltv["monetray_cltv_avg"] = df["total_spent"] / df["total_order"]

cltv = cltv[cltv["frequency"] > 0]

cltv.head()

################################### Establishing BG/NBD, Gamma-Gamma Models and Calculating CLTV  ###################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv["frequency"],cltv["recency_cltv_weekly"],cltv["T_weekly"])

cltv["exp_sales_3_month"] = bgf.predict(3*4,cltv["frequency"],cltv["recency_cltv_weekly"],cltv["T_weekly"])
cltv["exp_sales_6_month"] = bgf.predict(6*4,cltv["frequency"],cltv["recency_cltv_weekly"],cltv["T_weekly"])


ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(cltv["frequency"],cltv["monetray_cltv_avg"])

cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv["frequency"],cltv["monetray_cltv_avg"])

cltv["cltv"] = ggf.customer_lifetime_value(bgf,
                                           cltv["frequency"],
                                           cltv["recency_cltv_weekly"],
                                           cltv["T_weekly"],
                                           cltv["monetray_cltv_avg"],
                                           time=6,
                                           discount_rate=0.01,
                                           freq="W")


cltv.sort_values("cltv",ascending=False).head(20)



################################### Creating Segments Based on CLTV Value  ###################################

cltv["segment"]= pd.qcut(cltv["cltv"],4,labels=["D","C","B","A"])

cltv.sort_values("segment",ascending=False)

cltv.groupby("segment").agg({"cltv":["count","mean","sum"]})