######################################## Customer with RFM Analysis segmentation ########################################
import pandas as pd
import datetime as dt
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
pd.set_option("display.float_format",lambda x: "%.4f" % x)

df_ = pd.read_excel("online_retail_II.xlsx",sheet_name="Year 2010-2011")
df = df_.copy()


####################### Understanding and Preparing Data ######################

df.head()
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C",na=False)]

df["Monetray"] = df["Quantity"] * df["Price"]

df.describe().T
df = df[df["Price"] > 0]

####################### Calculation of RFM Metrics #######################

df["InvoiceDate"].max()
today = dt.datetime(2011,12,11)

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda x: (today - x.max()).days,
                               "Invoice": lambda x:x.nunique(),
                               "Monetray": lambda x: x.sum()})


rfm.columns = ["recency","frequency","monetary"]

####################### Calculation of RFM Score #######################

rfm["recency_metric"] = pd.qcut(rfm["recency"],5,labels=[5,4,3,2,1])
rfm["frequency_metric"] = pd.qcut(rfm["frequency"].rank(method="first"),5,labels=[1,2,3,4,5])
rfm["monetary_metric"] = pd.qcut(rfm["monetary"],5,labels=[1,2,3,4,5])

rfm["RF_SCORE"] = rfm["recency_metric"].astype(str) + rfm["frequency_metric"].astype(str)

####################### Defining RFM Score as a Segment #######################
seg_map = {r"[1-2][1-2]":"hibernating",
           r"[1-2][3-4]":"at_Risk",
           r"[1-2]5":"cant_loose",
           r"3[1-2]":"about_to_sleep",
           r"33":"need_attention",
           r"[3-4][4-5]":"loyal_customer",
           r"41":"promising",
           r"51":"new_customer",
           r"[4-5][2-3]":"potential_loyalists",
           r"5[4-5]":"champions"}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map,regex=True)

rfm.reset_index(inplace=True)

target_customer = rfm[rfm["segment"].str.contains("loyal_customer",na=False)]["Customer ID"]

####################### Printing in CSV format #######################

target_customer.to_csv("bonus_pr_I")
