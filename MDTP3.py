
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

total= pd.read_csv('MDTP3/total2014.csv')
eleitos= pd.read_csv('MDTP3/eleitos2014.csv')
candidatos = pd.merge(total, eleitos, how='left', on='nome')


# In[18]:


sns.pairplot(candidatos, hue='eleito')


# In[19]:


candidatos.to_csv('MDTP3/candidatos.csv')


# In[2]:


cand.head()


# In[3]:


cand = pd.read_csv('MDTP3/candidatos.csv')
cand.keys()


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


X = cand[[  'codigo', 'uf', 'partido',
       'quantidade_doacoes', 'quantidade_doadores', 'total_receita',
       'media_receita', 'recursos_de_outros_candidatos.comites',
       'recursos_de_pessoas_fisicas', 'recursos_de_pessoas_juridicas',
       'recursos_proprios', 'recursos_de_partido_politico',
       'quantidade_despesas', 'quantidade_fornecedores', 'total_despesa',
       'media_despesa', 'sexo',  'estado_civil']]

y = cand['eleito']


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


# In[7]:


from sklearn.svm import SVC
model = SVC()


# In[8]:


model.fit(X_train, y_train)



# In[9]:


predictions = model.predict(X_test)


# In[10]:


from sklearn.metrics import classification_report, confusion_matrix


# In[11]:


print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))


# In[12]:



from sklearn.model_selection import GridSearchCV


# In[13]:


param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}


# In[14]:


grid = GridSearchCV(SVC(),param_grid, verbose=3)


# In[105]:


grid.fit(X_train, y_train)


# In[106]:


grid.best_params_


# In[107]:


grid.best_estimator_


# In[108]:


grid_predictions = grid.predict(X_test)


# In[109]:


print(confusion_matrix(y_test, grid_predictions))
print('\n')
print(classification_report(y_test, grid_predictions))


# In[15]:


from sklearn.tree import DecisionTreeClassifier


# In[16]:


dtree = DecisionTreeClassifier()


# In[17]:


dtree.fit(X_train, y_train)


# In[18]:


predictions = dtree.predict(X_test)


# In[19]:


from sklearn.metrics import classification_report, confusion_matrix


# In[20]:


print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))


# In[21]:


from sklearn.ensemble import RandomForestClassifier


# In[31]:


rfc = RandomForestClassifier(n_estimators=1000)
rfc.fit(X_train, y_train)


# In[32]:


rfc_pred = rfc.predict(X_test)


# In[36]:


print(confusion_matrix(y_test,rfc_pred))
print('\n')
print(classification_report(y_test, rfc_pred))


# In[37]:


cand['eleito'].value_counts()


# In[38]:


print(pd.crosstab(y_test, rfc_pred,rownames=['real'], colnames=['Predito'],margins=True))


# In[39]:


# modelo classificou 1603 instancias que não foram eleitos corretamente
# então classificou 51 como eleitos que na verdade não foram eleitos
# classificou 53 como não eleitos que na verdade foram eleitos
# e 130 que foram eleitos como eleitos


# In[40]:


#para visualizacao da arvore
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot


# In[41]:


features = list(cand.columns[1:])
features


# In[42]:


dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, feature_names=features,filled=True, rounded = True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())


# In[145]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
dataset = cand
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
df


# In[43]:


# lista of colunas dataset
cols = cand.columns

rank = rfc.feature_importances_

features_dict = dict(zip(np.argsort(rank),cols))


# In[48]:


print (features_dict, "\n")


# In[63]:


importances = pd.DataFrame({'Caracteristica  ':X_train.columns,'Importância':np.round(rfc.feature_importances_,3)})
importances = importances.sort_values('Importância',ascending=False).set_index('Caracteristica  ')


# In[64]:


importances.head(18)


# In[65]:


importances.plot.bar()

