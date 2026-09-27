import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_customer = pd.read_excel('customer_churn_data_raw.xlsx', sheet_name='db_customer')
print(df_customer.head())
print(df_customer.info())
print(df_customer.rename(columns = {'name' : 'customer_name'}, inplace= True))
print(df_customer.drop(columns=['interests', 'pincode'], inplace=True))
print(df_customer.info())
df_customer['gender'] = df_customer['gender'].replace({'Men' : 'Male', 'Women' : 'Female'})
print(df_customer['gender'].value_counts())
print(df_customer[df_customer['country'].isna()])
state_country_mapping = df_customer.dropna(subset=['country']).set_index('state')['country'].to_dict()
df_customer['country'] = df_customer['country'].fillna(df_customer['state'].map(state_country_mapping))
print(df_customer[df_customer['country'].isna()])

print("=========================================================================")
df_subscription = pd.read_excel('customer_churn_data_raw.xlsx', sheet_name='db_subscription')
print(df_subscription.head())
print(df_subscription.info())
date_colums = ['subscription_start_date', 'renewal_date', 'cancellation_date']
df_subscription[date_colums] = df_subscription[date_colums].apply(pd.to_datetime)
print(df_subscription.info())

print("=========================================================================")
df_support = pd.read_excel('customer_churn_data_raw.xlsx', sheet_name='db_support')
print(df_support.head())
print(df_support.drop(columns = ['col_1', 'comment'], inplace = True))
print(df_support.info())

# feature engineering and data analysis

df_subscription["churn_flg"] = np.where(df_subscription['cancellation_date'].notnull(), 1, 0)
print(df_subscription.head())

# first fix support table duplicates then merge
# Note: while merging df's always check the shape before and after

df = pd.merge(df_customer, df_subscription, on='customerid', how='left').merge(df_support, on='customerid', how='left')
print(df)
print(df.columns)
print(df.shape)
print('df_subscription unique value:', df_subscription['customerid'].nunique())
print('df_customer unique value:', df_customer['customerid'].nunique())
print('df_support unique value:', df_support['customerid'].nunique())
print('df_support all value:', df_support['customerid'].size)

df_support['complaint_count'] = df_support.groupby('customerid')['customerid'].transform('count')
df_support = df_support.sort_values('complaint_date').drop_duplicates('customerid', keep= 'last')
print(df_support['customerid'].size)

# merge df
df = (df_subscription
            .merge(df_customer, on = 'customerid', how= 'left')
            .merge(df_support, on = 'customerid', how= 'left') )
print(df.shape)
print(df.columns)

# 1. Churn Rate
churn_rate = df['churn_flg'].mean()*100
print("Churn Rate = ", round(churn_rate,2), "%")

# 2. Retenion Rate
retention_rate = 100 - churn_rate
print("Retention rate = ", round(retention_rate,2), "%")

# 3. Churn by Plan type
churn_by_plan = df.groupby('plan_type')['churn_flg'].mean() * 100
print("Churn by Plan Type:",round(churn_by_plan,2), "%")

# 4 a. Churn by state + sum(revenue) & count of users
print(df.groupby("state")["monthly_charges"].sum())
print(df.groupby("state")["customerid"].count())

# 4 b. Churn by subscription type + sum(revenue) & count of users
print(df.groupby("subscription_type")["monthly_charges"].sum())
print(df.groupby("subscription_type")["customerid"].count())

# 5. ARPU - Avg Revenue per user
arpu = df['monthly_charges'].mean()
print("ARPU = ", round(arpu,2))

# 6. Revenue at risk - revenue lost from churned users
revenue_at_risk = df.loc[df['churn_flg']==1,'monthly_charges'].sum()
print("Revenue at Risk (Rs 'K') =", revenue_at_risk)

# 7. Esclation Rate
escalation_rate = (df['escalations']=='Y').mean()*100
print("Esclation Rate = ", round(escalation_rate, 2), "%")

# 8. Avg Complaint Per User
avg_complaints = df['complaint_count'].sum() / df['customerid'].nunique()
print("Avg Compliants Per User = ", round(avg_complaints, 2))

# 9. Correlation Esclation vs Churn
df['escalations'] = np.where(df['escalations'] == 'Y', 1, 0) # encoding string to int type
corr_df = df[['escalations', 'churn_flg']].dropna()
correlation = corr_df['escalations'].corr(df['churn_flg'])
print("Correlation between esclation vs churn is = ", round(correlation,2))

# 10. Create a column using existing col - Churn risk
conditions = [(df['churn_score'] < 50),
              (df['churn_score'] >= 50) & (df['churn_score'] < 70),
              (df['churn_score'] >= 70)]
choices = ['low', 'med', 'high']
df['churn_risk'] = np.select(conditions, choices, default='unknown')


# 4.Visualization using Matplotlib
print(df.columns)

# 4.1 Monthly Churn Trend (Time Series KPI)
df['cancellation_month'] = df['cancellation_date'].dt.to_period ('M')
churn_trend = df[df['churn_flg'] == 1].groupby('cancellation_month').size()
plt.figure(figsize=(8,4))
plt.plot(
    churn_trend.index.astype(str).to_numpy(),
    churn_trend.to_numpy(dtype=float),
    color='green',
    marker='o',
    linestyle='--',
    linewidth=2,
    markersize=12,
)

plt.title('Monthly Churn Trend')
plt.xlabel('Month')
plt.ylabel('Churned Customers')
plt.show()

# 4.2 Churn Rate by Plan type
churn_plan = df.groupby('plan_type')['churn_flg'].mean()

colors = plt.cm.Set2(np.linspace(0, 1, len(churn_plan)))
plt.figure(figsize=(7,4))
plt.bar(churn_plan.index.astype(str), churn_plan.to_numpy(dtype=float), color=colors)
plt.title('Churn Rate by Plan Type')
plt.xlabel('Plan Type')
plt.ylabel('Churn Rate (%)')
plt.show()


# 4.3 Churn by States
churn_plan = df.groupby('state')['churn_flg'].mean()

colors = plt.cm.Set2(np.linspace(0, 1, len(churn_plan)))
plt.figure(figsize=(12,6))
plt.bar(churn_plan.index.astype(str), churn_plan.to_numpy(dtype=float), color=colors)
plt.title('Churn Rate by State')
plt.xlabel('State')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=10)
plt.show()

