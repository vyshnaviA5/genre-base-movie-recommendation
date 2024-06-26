#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv(r"C:\Users\VivoBook\Downloads\movie\ratings.csv")
df


# In[2]:


df1=pd.read_csv(r"C:\Users\VivoBook\Downloads\movie\ratings_small.csv")
df1


# In[3]:


df2=pd.read_csv(r"C:\Users\VivoBook\Downloads\movie\movies_metadata.csv")
df2


# In[4]:


df3=pd.read_csv(r"C:\Users\VivoBook\Downloads\movie\credits.csv")
df3


# In[5]:


df4=pd.read_csv(r"C:\Users\VivoBook\Downloads\movie\keywords.csv")
df4


# In[6]:


df5=pd.read_csv(r"C:\Users\VivoBook\Downloads\movie\links_small.csv")
df5


# In[7]:


df6=pd.read_csv(r"C:\Users\VivoBook\Downloads\movie\links.csv")
df6


# In[8]:


main=pd.read_csv(r"C:\Users\VivoBook\Downloads\movie\movies_metadata.csv")
credits=pd.read_csv(r"C:\Users\VivoBook\Downloads\movie\credits.csv")
keywords=pd.read_csv(r"C:\Users\VivoBook\Downloads\movie\keywords.csv")


# In[9]:


main.head()


# In[10]:


main.tail()


# In[11]:


credits.head()


# In[12]:


credits.tail()


# In[13]:


keywords.head()


# In[14]:


keywords.tail()


# In[15]:


data=pd.concat([main,credits,keywords],axis='columns')
data


# In[16]:


data[data.duplicated()]


# In[17]:



data=data.drop_duplicates()


# In[18]:


data.columns


# In[20]:


data=data[["id","title","overview","genres","keywords","cast","crew"]]


# In[21]:


data=data.iloc[:10000,2:]
data.head(2)


# In[22]:


data.isnull().sum()


# In[23]:


data=data.dropna()
data.isnull().sum()


# In[24]:


import ast
def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l    


# In[25]:


data["genres"]=data["genres"].apply(convert)
data['keywords']=data["keywords"].apply(convert)


# In[26]:


def convert3(obj):
    l=[]
    c=0
    for i in ast.literal_eval(obj):
        if c!=3:
            l.append(i['name'])
        else:
            break    
    return l    


# In[27]:


data["cast"]=data["cast"].apply(convert3)


# In[28]:


def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i["job"]=="Director":
            l.append(i['name'])
            break
    return l    


# In[29]:


data["crew"]=data["crew"].apply(fetch_director)


# In[30]:


data.head(3)


# In[31]:


data["overview"]=data["overview"].apply(lambda x:x.split())


# In[32]:


data["genres"]=data["genres"].apply(lambda x:[i.replace(" ","") for i in x])
data['keywords']=data['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
data["cast"]=data['cast'].apply(lambda x:[i.replace(" ","") for i in x])
data['crew']=data['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[33]:


data.head(2)


# In[34]:


data["tag"]=data['overview']+data['genres']+data['keywords']+data["cast"]+data['crew']
new_df=data[["id","title","tag"]]
new_df.head()


# In[35]:


new_df["tag"]=new_df["tag"].apply(lambda x:" ".join(x))


# In[36]:


new_df["tag"]=new_df["tag"].apply(lambda x:x.lower())


# In[37]:


new_df.head()


# In[38]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[39]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)    


# In[40]:


new_df["tag"]=new_df["tag"].apply(stem)


# In[41]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=20000,stop_words='english')


# In[42]:


vectores=cv.fit_transform(new_df["tag"]).toarray()


# In[43]:


vectores.shape


# In[44]:


from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectores)


# In[45]:


similarity.shape


# In[46]:


def recom(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    dist=similarity[movie_index]
    movie_list=sorted(list(enumerate(dist)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# In[47]:


recom('Toy Story')


# In[48]:


director=[ ]
for i in new_df["tag"]:
    j=i.split()[-1]
    director.append(j)


# In[49]:


q=pd.DataFrame(director)


# In[50]:


def recom2(director):
    director_list=q[q[0]==director].index[:5]

    for i in director_list:
        print(new_df.iloc[i].title)


# In[51]:


recom2("woodyallen")


# In[ ]:




