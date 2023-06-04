#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import library 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,SpectralClustering
from sklearn.model_selection import train_test_split


# In[3]:


loan = pd.read_csv('loan_data_2007_2014.csv')
copy = loan.copy()
copy.info()


# In[4]:


# melakukan drop kolom 
copy.drop(['annual_inc_joint','dti_joint','verification_status_joint','open_acc_6m','open_il_6m','open_il_12m',
           'open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m', 'max_bal_bc',
           'all_util', 'inq_fi','total_cu_tl','inq_last_12m','Unnamed: 0','id','member_id','desc','mths_since_last_delinq',
           'mths_since_last_record','next_pymnt_d','mths_since_last_major_derog','url'], axis=1, inplace=True)


# In[101]:


copy.info()


# In[102]:


# memeriksa missing value
copy.isnull().sum()


# In[103]:


modus = copy['tot_coll_amt'].mode()[0]
copy['acc_now_delinq'].fillna(modus, inplace = True)
copy['collections_12_mths_ex_med'].fillna(modus, inplace=True)
copy['tot_coll_amt'].fillna(modus, inplace=True)


# In[104]:


modus = copy['delinq_2yrs'].mode()[0]
copy['delinq_2yrs'].fillna(modus,inplace=True)


# In[105]:


mean= copy['tot_cur_bal'].mean()
copy['tot_cur_bal'].fillna(mean, inplace = True)
copy['total_rev_hi_lim'].fillna(mean, inplace= True)


# In[106]:


copy['total_acc'].fillna(mean, inplace=True)


# In[107]:


copy['revol_util'].fillna(mean, inplace =True)


# In[108]:


copy['inq_last_6mths'].fillna(modus, inplace=True)


# In[109]:


median = copy['pub_rec'].median()
median1 =copy['open_acc'].median()
copy['pub_rec'].fillna(median, inplace=True)
copy['open_acc'].fillna(median, inplace=True)


# In[110]:


copy['annual_inc'].fillna(median, inplace=True)


# In[111]:


copy['term'] = copy['term'].str.replace(' months','')


# In[112]:


# merubah type data
copy['term'] = copy['term'].astype(int)


# In[113]:


# menghilangkan months pada kolom term
copy['emp_length'] = copy['emp_length'].str.extract('(\d+)')


# In[114]:


copy['emp_length'].fillna(0, inplace=True)
copy['emp_length'] = copy['emp_length'].astype(int)
copy['emp_length'] = copy['emp_length'].astype(float).astype(int)


# In[115]:


# mengambil bulannya saja pada kolom issue_d
copy['priode_pinjaman'] = copy['issue_d'].str.extract('([a-zA-Z]+)', expand=False)


# In[116]:


copy['emp_length'].fillna(median, inplace=True)


# In[77]:


# memperbaikin missing value pada kolom title
copy.dropna(subset=['title'], inplace=True)


# In[78]:


# memperbaikin missing value pada kolom emp_title
copy.dropna(subset=['emp_title'],inplace=True)


# In[79]:


# menghilangkan angka pada kolom-kolom tanggal 
copy['Earliest_cr_line'] = copy['earliest_cr_line'].str.extract('([a-zA-Z]+)', expand=False)
copy['Last_pymnt_d'] = copy['last_pymnt_d'].str.extract('([a-zA-Z]+)', expand=False)
copy['Last_credit_pull_d'] = copy['last_credit_pull_d'].str.extract('([a-zA-Z]+)', expand=False)


# In[80]:


copy.isnull().sum()


# In[81]:


# mendrop kolom yang sudah di perbaiki
copy.drop(['earliest_cr_line','last_pymnt_d','last_credit_pull_d'], axis=1, inplace=True)


# In[82]:


copy.dropna(subset=['Earliest_cr_line','Last_pymnt_d','Last_credit_pull_d'],inplace=True)


# In[83]:


copy[['Earliest_cr_line','Last_pymnt_d','Last_credit_pull_d']].sample(10)


# In[98]:


copy.head()


# In[ ]:


# untuk melakukan save
df.to_csv('VIX_IDX_Revisi.csv', index=False)


# In[84]:


fig, ax = plt.subplots(1, 1, figsize = (10,5))

sns.kdeplot(data = copy, x = df['loan_amnt'], alpha = 1, multiple = 'stack', hue = 'grade')
ax.set_title('Distribution of Loan by Grade', fontsize = 18)
ax.set_xlabel('Loan Amnt')


# In[85]:


plt.figure(figsize=(8, 6))
plt.hist(copy['int_rate'], bins=5, color='skyblue')
plt.xlabel('Suku Bunga')
plt.ylabel('Jumlah Pinjaman')
plt.title('Histogram Suku Bunga')
plt.show()


# In[86]:


ab=['total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee']
plt.figure(figsize=(12,8))
for i in range(len(ab)):
    plt.subplot(2, 3, i+1)
    sns.kdeplot(x=copy[ab[i]],color='grey')
    plt.xlabel(ab[i])
    plt.tight_layout()


# In[87]:


plt.figure(figsize=(10,15))
for i in range(0, len(ab)):
    plt.subplot(len(ab),3, i+1)
    sns.boxplot(y = copy[ab[i]], orient='v')
    plt.tight_layout()


# In[88]:


plt.figure(figsize=(8, 6))
copy['grade'].value_counts().plot(kind='bar', color='orange')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.title('Bar Plot - Loan Grade')
plt.show()


# In[90]:


plt.figure(figsize=(8, 6))
plt.boxplot(copy['int_rate'])
plt.ylabel('Interest Rate')
plt.title('Box Plot - Interest Rate')
plt.show()


# In[91]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='grade', y='loan_amnt', data=copy)
plt.xlabel('Grade')
plt.ylabel('Loan Amount')
plt.title('Box Plot - Loan Amount by Grade')
plt.show()


# In[92]:


import matplotlib.pyplot as plt

grade_counts = copy['grade'].value_counts().sort_index()

# Menghitung persentase rata-rata untuk setiap nilai
grade_percentages = (grade_counts / len(copy)) * 100

# Membuat visualisasi
fig, ax = plt.subplots()

# Mengatur warna batang berdasarkan tingkat persentase
colors = ['red' if p < 10 else 'blue' if p < 20 else 'green' for p in grade_percentages]
ax.bar(grade_percentages.index, grade_percentages, color=colors)

plt.xlabel('Grade')
plt.ylabel('Persentase')
plt.title('Rata-rata Persentase untuk Kolom "Grade"')

# Menambahkan angka di atas batang
for i, v in enumerate(grade_percentages):
    ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

# Mengatur label huruf abjad
custom_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
ax.set_xticklabels(custom_labels)

plt.show()


# In[117]:


import calendar
fig, ax = plt.subplots(figsize=(8, 6))

# Menghitung jumlah kemunculan setiap nilai priode_pinjaman
loan_counts = copy['priode_pinjaman'].value_counts()

# Mengurutkan indeks berdasarkan urutan bulan
sorted_months = sorted(loan_counts.index, key=lambda x: list(calendar.month_abbr).index(x))

# Membuat visualisasi grafik
ax.plot(sorted_months, loan_counts[sorted_months].values, color='blue', marker='o')

plt.xlabel('Periode Pinjaman')
plt.ylabel('Jumlah')
plt.title('Jumlah Pinjaman berdasarkan Periode Pinjaman')

# Menambahkan label jumlah di atas titik
for i, v in enumerate(loan_counts[sorted_months].values):
    ax.text(i, v, str(v), ha='center', va='bottom')

plt.show()


# In[94]:


# Menghitung count untuk setiap nilai term
term_counts = copy['term'].value_counts()

# Menghitung persentase rata-rata untuk setiap nilai
term_percentages = (term_counts / len(copy)) * 100

# Membuat visualisasi
ax = term_percentages.plot(kind='bar', color=['red', 'blue'])

plt.xlabel('Term')
plt.ylabel('Persentase')
plt.title('Rata-rata Persentase untuk Kolom "Term"')

# Menambahkan angka di atas batang
for i, v in enumerate(term_percentages):
    ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

plt.show()


# In[95]:


fig, ax = plt.subplots(figsize=(8, 6))

# Menghitung jumlah kemunculan setiap nilai home_ownership
ownership_counts = copy['home_ownership'].value_counts()

# Mengurutkan indeks berdasarkan jumlah secara terbalik
sorted_ownership = sorted(ownership_counts.index, key=lambda x: ownership_counts[x], reverse=True)

# Membuat visualisasi bar chart
ax.bar(sorted_ownership, ownership_counts[sorted_ownership], color='blue')

plt.xlabel('Home Ownership')
plt.ylabel('Jumlah')
plt.title('Jumlah Berdasarkan Home Ownership')

# Menambahkan label jumlah di atas batang
for i, v in enumerate(ownership_counts[sorted_ownership]):
    ax.text(i, v, str(v), ha='center', va='bottom')

plt.show()


# In[10]:


# Menghitung jumlah kemunculan setiap nilai purpose
purpose_counts = copy['purpose'].value_counts()

# Menghitung persentase untuk setiap nilai purpose
purpose_percentages = (purpose_counts / len(copy)) * 100

# Membuat visualisasi bar chart dengan seaborn
plt.figure(figsize=(10,7))
sns.barplot(x=purpose_percentages.index, y=purpose_percentages.values, color='blue')

plt.xlabel('Purpose')
plt.ylabel('Persentase')
plt.title('Persentase Berdasarkan Purpose')

# Menambahkan label persentase di atas batang
for i, v in enumerate(purpose_percentages.values):
    plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

plt.xticks(rotation='vertical')  # Mengubah orientasi label purpose menjadi vertical

plt.show()


# In[97]:


# Menghitung count untuk setiap nilai loan_status
loan_status_counts = copy['loan_status'].value_counts()

# Menghitung persentase rata-rata untuk setiap nilai
loan_status_percentages = (loan_status_counts / len(copy)) * 100

# Membuat visualisasi
ax = loan_status_percentages.plot(kind='bar', color=['red', 'blue'])

plt.xlabel('Loan Status')
plt.ylabel('Persentase')
plt.title('Rata-rata Persentase untuk Kolom "Loan Status"')

# Menambahkan angka di atas batang
for i, v in enumerate(loan_status_percentages):
    ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

plt.show()


# # Permodelan

# In[2]:


# import dataset untuk permodelan
data =pd.read_csv('VIX_IDX_Revisi.csv')
original_feature = data.columns
print('Jumlah Kolom',len(original_feature))
print(data.head())


# In[ ]:


pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
data.head()


# # Feature Selection

# In[3]:


# melakukan drop pada kolom yang tidak di perlukan secara manual
data.drop(['funded_amnt','sub_grade','emp_title','home_ownership','purpose','title',
          'zip_code','addr_state','initial_list_status','policy_code','application_type','priode_pinjaman',
          'Earliest_cr_line','Last_pymnt_d','Last_credit_pull_d','pymnt_plan'], axis=1, inplace=True)
print(data.info())


# # Label Encoding

# In[4]:


mapp_ver ={
     'Source Verified': 1, 
    'Not Verified': 0, 
    'Verified': 2
}
data['verification_status'] = data['verification_status'].map(mapp_ver)


# In[5]:


mapping_loann ={
    'Charged Off': 0, 
    'Fully Paid' : 1, 
    'Current' : 2, 
    'Default' : 5,
    'Late (31-120 days)' : 4, 
    'In Grace Period' : 3, 
    'Late (16-30 days)' : 6,
    'Does not meet the credit policy. Status:Fully Paid' : 7,
    'Does not meet the credit policy. Status:Charged Off' : 8
}
data['loan_status'] =data['loan_status'].map(mapping_loann)


# In[7]:


# melakukan label encod
mapping_grade ={
    'A' : 1,
    'B' : 2,
    'C' : 3,
    'D' : 4,
    'E' : 5,
    'F' : 6,
    'G' : 7
}
data['grade'] = data['grade'].map(mapping_grade)


# In[8]:


data['loan_status'].value_counts()


# In[9]:


fil = (data['loan_status'] == 8 ) | (data['loan_status'] == 5 )
data.loc[fil, 'loan_status'] = 0
data['loan_status'].value_counts()


# In[10]:


fill =(data['loan_status']==7)
data.loc[fill, 'loan_status']=1
data['loan_status'].value_counts()


# In[11]:


fill =(data['loan_status']==6) 
data.loc[fill, 'loan_status']=4
data['loan_status'].value_counts()


# # Handle Outlier With Z-score

# In[13]:


def count_outliers(column):
    if column.dtype.kind in ['i', 'f']:
        q1 = np.percentile(column, 25)
        q3 = np.percentile(column, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = column[(column < lower_bound) | (column > upper_bound)]
        return len(outliers)
    else:
        return 0

def calculate_outlier_percentage(column):
    total_data = len(column)
    outliers = count_outliers(column)
    percentage = (outliers / total_data) * 100
    return percentage

outlier_counts = data.apply(count_outliers)
outlier_percentages = data.apply(calculate_outlier_percentage)

print("Jumlah Outlier:")
print(outlier_counts)
print("\nPersentase Outlier:")
print(outlier_percentages)


# In[23]:


plt.figure(figsize=(35,20))
data.boxplot()
plt.title('Deteksi Outlier')
plt.ylabel('Nilai')
plt.xticks(rotation ='vertical')
plt.show()


# In[16]:


cols = ['installment', 'int_rate', 'open_acc', 'total_acc', 'tot_cur_bal', 'last_pymnt_amnt', 'revol_bal',
        'annual_inc', 'out_prncp_inv', 'out_prncp', 'total_rec_prncp', 'total_rec_int', 'total_pymnt',
        'total_pymnt_inv']
z_scores = np.abs(stats.zscore(data[cols]))
                                     
filtered_entries = (z_scores < 3).all(axis=1)
handle_out=data[filtered_entries]
handle_out.describe()


# # Korelasi

# In[19]:


handle_out.corr()


# In[20]:


# mengambil 18 feature terbaik 
corrmat = handle_out.corr()
k = 18
cols = corrmat.nlargest(k, corrmat.columns).index
cm = np.corrcoef(data[cols].values.T)
# Membuat visulisasi 
sns.set(font_scale=1.25)
plt.subplots(figsize=(20,20))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()


# In[22]:


df = handle_out[['loan_amnt', 'installment','funded_amnt_inv','grade','int_rate','open_acc','total_acc','tot_cur_bal','last_pymnt_amnt','revol_bal',
                 'annual_inc','verification_status','term','out_prncp_inv','out_prncp','total_rec_prncp','total_rec_int',
                 'total_pymnt','total_pymnt_inv']].reset_index()
df.info()
df = df.drop('index', axis=1)


# In[27]:


df.info()


# # Standarisasi

# In[28]:


df_s = df
scalar=StandardScaler()
scaler_df = scalar.fit_transform(df_s)
scaler_df = pd.DataFrame(scaler_df,columns=df_s.columns)
scaler_df.head()


# In[30]:


pca = PCA (n_components=2)
principal_components = pca.fit_transform(scaler_df)
pca_df = pd.DataFrame(data=principal_components, columns=['PCA1','PCA2'])
pca_df


# # Elbow Method

# In[31]:


inertia = []
range_val = range(1, 18)
random_state = 42  # Nilai random_state yang digunakan
for i in range_val:
    kmean = KMeans(n_clusters=i, random_state=random_state)  # Menambahkan random_state
    kmean.fit_predict(pd.DataFrame(scaler_df))
    inertia.append(kmean.inertia_)
plt.plot(range_val, inertia, 'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show()


# # KMeans

# In[32]:


kmeans_model = KMeans(n_clusters=3, random_state=42)
kmeans_model.fit_predict(scaler_df) 
pca_df_kmeans= pd.concat([pca_df, pd.DataFrame({'cluster':kmeans_model.labels_})],axis=1)
pca_df_kmeans.sample(10)


# ## Visualisasi Df Cluster

# In[33]:


plt.figure(figsize =(8,8))
ax= sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data= pca_df_kmeans, palette=['yellow','green','blue'])
plt.title('Clustering using K-Means Algorithm')
plt.show()


# In[35]:


df.info()


# In[36]:


cluster_center = pd.DataFrame(data = kmeans_model.cluster_centers_,columns=[scaler_df.columns])
cluster_center = scalar.inverse_transform(cluster_center)
cluster_center = pd.DataFrame(data =cluster_center, columns=[df.columns])
cluster_center


# In[38]:


cluster_df = pd.concat([df,pd.DataFrame({'cluster':kmeans_model.labels_})],axis=1)
cluster_df.head()


# In[39]:


# cek cluster 0
cluster_df[cluster_df['cluster']==0]. head()


# In[40]:


# cluster 1
cluster_df[cluster_df['cluster']==1].head()


# In[41]:


# cluster2
cluster_df[cluster_df['cluster']==2].head()


# In[42]:


sns.countplot(x='cluster',data= cluster_df)


# In[43]:


for i in cluster_df.drop(['cluster'],axis=1):
    grid = sns.FacetGrid(cluster_df, col='cluster')
    grid = grid.map(plt.hist, i)
plt.show()


# In[44]:


#Saving Scikitlearn models
import joblib
joblib.dump(kmeans_model, "kmeans_model.pkl")


# In[46]:


cluster_df.to_csv("Clustered_Customer_Data_1.csv")


# # Training and Testing the Model 

# In[47]:


cluster_df.info()


# In[48]:


# split dataset
X = cluster_df.drop(['cluster'], axis=1)
y = cluster_df[['cluster']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[50]:


X_train


# In[51]:


X_test


# In[52]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)

# melakukan prediksi pada data uji 
y_pred = model_LR.predict(X_test)

# Evaluasi model 
confusion_mat = confusion_matrix(y_test,y_pred)
classification_rep = classification_report(y_test,y_pred)

# menampilkan hasil output
print('Confusian Matrik: ')
print(confusion_mat)

print('Classification Report:')
print(classification_rep)


# In[53]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Inisialisasi model dengan hyperparameter yang diinginkan
model_DT = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=2)

# Melatih model
model_DT.fit(X_train, y_train)

# Melakukan prediksi pada data uji
y_pred = model_DT.predict(X_test)

# Evaluasi model
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Menampilkan hasil output
print("Confusion Matrix:")
print(confusion_mat)
print("Classification Report:")
print(classification_rep)


# In[55]:


import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Inisialisasi model dengan hyperparameter yang diinginkan
model_xgb = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, subsample=0.8, colsample_bytree=0.8)

# Melatih model
model_xgb.fit(X_train, y_train)

# Melakukan prediksi pada data uji
y_pred = model_xgb.predict(X_test)

# Evaluasi model
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Menampilkan hasil output
print("Confusion Matrix:")
print(confusion_mat)
print("Classification Report:")
print(classification_rep)


# In[56]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Inisialisasi model dengan hyperparameter yang diinginkan
model_RF = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=2)

# Melatih model
model_RF.fit(X_train, y_train)

# Melakukan prediksi pada data uji
y_pred = model_RF.predict(X_test)

# Evaluasi model
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Menampilkan hasil output
print("Confusion Matrix:")
print(confusion_mat)
print("Classification Report:")
print(classification_rep)


# # Saving XGboost model for future prediction

# In[58]:


import pickle
filename = 'final_model.sav'
pickle.dump(model_xgb, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result,'% Acuracy')


# # Go to Deploy Model ======== >
