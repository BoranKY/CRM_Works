######################################## Customer with RFM Analysis segmentation ########################################

import pandas as pd
import datetime as dt
pd.set_option("display.max_columns",None)
#pd.set_option("display.max_rows",None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

####################### Understanding and Preparing Data #######################
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()


df.head()
df.columns
df.describe().T
df.isnull().sum()
df.info()

df[ "total_order" ] = df[ "order_num_total_ever_online" ] + df[ "order_num_total_ever_offline" ]
df[ "total_spend" ] = df[ "customer_value_total_ever_online" ] + df[ "customer_value_total_ever_offline" ]

category = [ "first_order_date","last_order_date","last_order_date_online","last_order_date_offline" ]
for cat in category:
    df[ cat ] = pd.to_datetime(df[ cat ])

df.groupby("order_channel").agg({"master_id":lambda x:x.nunique(),
                                    "total_order":"sum",
                                     "total_spend":"sum"})

df.sort_values(by="total_spend",ascending=False).head(10)
df.sort_values(by="total_order",ascending=False).head(10)

####################### FUNCTION #######################
def general_info(dataframe):
    dataframe[ "total_order" ] = dataframe[ "order_num_total_ever_online" ] + dataframe[ "order_num_total_ever_offline" ]
    dataframe[ "total_spend" ] = dataframe[ "customer_value_total_ever_online" ] + dataframe[ "customer_value_total_ever_offline" ]

    category = [ "first_order_date","last_order_date","last_order_date_online","last_order_date_offline" ]
    for cat in category:
        dataframe[ cat ] = pd.to_datetime(df[ cat ])
    return dataframe

####################### Calculation of RFM Metrics #######################
today = dt.datetime(2021,6,1)

rfm = pd.DataFrame()
rfm["customer_id"]=df["master_id"]
rfm["Recency"] = (today - df["last_order_date"] )
rfm["Frequency"] = df["total_order"]
rfm["Monetary"] = df["total_spend"]

rfm.head()
rfm.shape

####################### Calculation of RF Score #######################

rfm["recency_score"] = pd.qcut(rfm["Recency"],5,[5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"),5,[1,2,3,4,5])
rfm["monetary_score"] = pd.qcut(rfm["Monetary"],5,[1,2,3,4,5])
rfm.head()

rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)



####################### Defining RF Score as a Segment #######################

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


rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

####################### Tasks #######################

rfm[["segment","Recency","Frequency","Monetary"]].groupby("segment").agg({"mean","count"})
rfm["segment"].value_counts()

# Loyal customers (champions, loyal_customers) and women shoppers will be contacted specifically.
# Save the ID numbers of these customers in the csv file.

target_customer= rfm[rfm["segment"].isin(["champions","loyal_customer"])]["customer_id"]
target_customer_id = df[(df["master_id"].isin(target_customer)) & (df["interested_in_categories_12"].str.contains("KADIN",na=False))]["master_id"]
target_customer_id.to_csv("target_customer_id.csv", index=False)

# "cant_loose","at_Risk","hibernating","new_customer" save the
# ids of the people in this profile and the customers in the profile to the csv file

target_segment = rfm[rfm["segment"].isin(["cant_loose","at_Risk","hibernating","new_customer"])]["customer_id"]
cust_ids =df[(df["master_id"].isin(target_segment)) & (df["interested_in_categories_12"].str.contains("ERKEK",na= False) | df["interested_in_categories_12"].str.contains("COCUK",na= False))]["master_id"]
cust_ids.to_csv("cust_ids.csv")