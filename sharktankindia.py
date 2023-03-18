import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import time
import base64



def shark(user_sentence):
 df1 = pd.read_excel('SharkTankIndiaS02.xlsx')
 df=df1[["Idea","Deal"]].copy()
 df["deal/no deal"]=df["Deal"].copy()
 df['deal/no deal']=df['deal/no deal'].apply(lambda x: 'Deal' if x != 'No Deal' else x)
 df['deal in binary']=df['deal/no deal'].apply(lambda x: 1 if x != 'No Deal' else x)
 df["deal in binary"] = df["deal in binary"].replace(['No Deal'],[0])
# Replacing punctuations with space
 df['Idea_processed'] = df['Idea'].str.replace("[^a-zA-Z0-9]", " ")
 df['Idea_processed'] = df['Idea_processed'].astype(str)
 df['Idea_processed'] = df['Idea_processed'].apply(lambda row: ' '.join([word for word in row.split() if len(word)>2]))
# make entire text lowercase
 df['Idea_processed'] = [row.lower() for row in df['Idea_processed']]

# Removing Stopwords Begin
 import nltk
 nltk.download('punkt')
 nltk.download('stopwords')
 from nltk.corpus import stopwords
 from nltk import word_tokenize

 stop_words = stopwords.words('english') # extracting all the stop words in english language and storing it in a variable called stop_words -> set

# Making custom list of words to be removed 
 add_words = ['brand','company']

# Adding to the list of words
 stop_words.extend(add_words)

# Function to remove stop words 
 def remove_stopwords(rev):
    # iNPUT : IT WILL TAKE ROW/REVIEW AS AN INPUT
    # take the paragraph, break into words, check if the word is a stop word, remove if stop word, combine the words into a para again
    review_tokenized = word_tokenize(rev)
    rev_new = " ".join([i for i in review_tokenized  if i not in stop_words])
    return rev_new

# Removing stopwords
 df['Idea_processed'] = [remove_stopwords(r) for r in df['Idea_processed']]
# Begin Lemmatization 
 nltk.download('wordnet')
 nltk.download('omw-1.4')
 nltk.download('averaged_perceptron_tagger')
 from nltk.stem import WordNetLemmatizer
 from nltk.corpus import wordnet

# function to convert nltk tag to wordnet tag
 lemmatizer = WordNetLemmatizer()

# Finds the part of speech tag
# Convert the detailed POS tag into a shallow information
 def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

# lemmatize sentence using pos tag
 def lemmatize_sentence(sentence):
  # word tokenize -> pos tag (detailed) -> wordnet tag (shallow pos) -> lemmatizer -> root word
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  # output will be a list of tuples -> [(word,detailed_tag)]
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged) # output -> [(word,shallow_tag)]
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


 df['Idea_processed'] = df['Idea_processed'].apply(lambda x: lemmatize_sentence(x))



# Importing module
 from sklearn.feature_extraction.text import TfidfVectorizer

# Creating matrix of top 2500 tokens
 tfidf = TfidfVectorizer(max_features=2500)
# tmp_df = tfidf.fit_transform(df.review_processed)
# feature_names = tfidf.get_feature_names()
# pd.DataFrame(tmp_df.toarray(), columns = feature_names).head() 

 X = tfidf.fit_transform(df.Idea_processed).toarray()
 y = df['deal/no deal'].map({'Deal' : 1, 'No Deal' : 0}).values
 featureNames = tfidf.get_feature_names_out()





# # Splitting the dataset into train and test
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

 from sklearn.tree import DecisionTreeClassifier

 dt = DecisionTreeClassifier()
 dt.fit(X_train,y_train)

 y_pred = dt.predict(X_test)

 user_sentence = user_sentence.replace("[^a-zA-Z0-9]", " ")
 user_sentence = ' '.join([x for x in user_sentence.split() if len(x) > 2])
 user_sentence = user_sentence.lower()

 user_sentence = remove_stopwords(user_sentence)
 user_sentence = lemmatize_sentence(user_sentence)


 X_test = tfidf.transform([user_sentence]).toarray()


 predictdeal= dt.predict(X_test)
 a= predictdeal[0]
 if a==1:
  b="your Idea seems good. you will get a Deal in sharktank 👌"
 else:
  b="you need to work more on your Idea.you will Not get a Deal in sharktank 👎"

 return b

def invest():
 df2 = pd.read_excel('SharkTankIndiaS02.xlsx')
 df3=df2[["Idea","Brand","Deal","Amit Jain","Namita","Anupam","Vineeta","Aman","Piyush"]].copy()
 df3["deal/no deal"]=df2["Deal"].copy()
 df3['deal/no deal']=df3['deal/no deal'].apply(lambda x: 'Deal' if x != 'No Deal' else x)
 df3["Amit Jain"] = df3["Amit Jain"].replace({'Amit':"Deal", "ABSENT":"No Deal", "NaN":"No Deal"})
 df3["Namita"] = df3["Namita"].replace({'Namita':"Deal", "ABSENT":"No Deal", "NaN":"No Deal"})
 df3["Anupam"] = df3["Anupam"].replace({'Anupam':"Deal", "ABSENT":"No Deal", "NaN":"No Deal"})
 df3["Vineeta"] = df3["Vineeta"].replace({'Vineeta':"Deal", "ABSENT":"No Deal", "NaN":"No Deal"})
 df3["Aman"] = df3["Aman"].replace({'Aman':"Deal", "ABSENT":"No Deal", "NaN":"No Deal"})
 df3["Piyush"] = df3["Piyush"].replace({'Piyush':"Deal","Peyush":"Deal", "ABSENT":"No Deal", "NaN":"No Deal"})
 d = [df3["Amit Jain"].value_counts(), df3["Namita"].value_counts(), df3["Anupam"].value_counts(),df3["Vineeta"].value_counts(), df3["Aman"].value_counts(), df3["Piyush"].value_counts()]
 c=pd.DataFrame(d)
 c.reset_index(inplace = True)

 return c

def interest():
 df4 = pd.read_excel('SharkTankIndiaS02.xlsx')
 df5=df4[["Idea","Brand","Deal","Amit Jain","Namita","Anupam","Vineeta","Aman","Piyush"]].copy()
 df5["Amit Jain"] = df5["Amit Jain"].replace({'Amit':"Deal", "ABSENT":"No Deal", "NaN":"No Deal"})
 df5["Namita"] = df5["Namita"].replace({'Namita':"Deal", "ABSENT":"No Deal", "NaN":"No Deal"})
 df5["Anupam"] = df5["Anupam"].replace({'Anupam':"Deal", "ABSENT":"No Deal", "NaN":"No Deal"})
 df5["Vineeta"] = df5["Vineeta"].replace({'Vineeta':"Deal", "ABSENT":"No Deal", "NaN":"No Deal"})
 df5["Aman"] = df5["Aman"].replace({'Aman':"Deal", "ABSENT":"No Deal", "NaN":"No Deal"})
 df5["Piyush"] = df5["Piyush"].replace({'Piyush':"Deal","Peyush":"Deal", "ABSENT":"No Deal", "NaN":"No Deal"})
 df6=df5[["Idea",option1,option2]]
 df7= df6.loc[(df6[option1]=="Deal")&(df6[option2]=="Deal")]
 df7["investor"]= df7[option1]

 return df7

def data():
 df8 = pd.read_excel('SharkTankIndiaS02.xlsx')
 df9=df8[["Idea","Deal","Amit Jain","Namita","Anupam","Vineeta","Aman","Piyush"]].copy()
 df9["deal/no deal"]=df9["Deal"].copy()
 df9['deal/no deal']=df9['deal/no deal'].apply(lambda x: 'Deal' if x != 'No Deal' else x)
 df9['deal in binary']=df9['deal/no deal'].apply(lambda x: 1 if x != 'No Deal' else x)
 df9["deal in binary"] = df9["deal in binary"].replace(['No Deal'],[0])
# Replacing punctuations with space
 df9['Idea_processed'] = df9['Idea'].str.replace("[^a-zA-Z0-9]", " ")
 df9['Idea_processed'] = df9['Idea_processed'].astype(str)
 df9['Idea_processed'] = df9['Idea_processed'].apply(lambda row: ' '.join([word for word in row.split() if len(word)>2]))
# make entire text lowercase
 df9['Idea_processed'] = [row.lower() for row in df9['Idea_processed']]
 # Removing Stopwords Begin
 import nltk
 nltk.download('punkt')
 nltk.download('stopwords')
 from nltk.corpus import stopwords
 from nltk import word_tokenize

 stop_words = stopwords.words('english') # extracting all the stop words in english language and storing it in a variable called stop_words -> set

# Making custom list of words to be removed 
 add_words = ['brand','company']

# Adding to the list of words
 stop_words.extend(add_words)

# Function to remove stop words 
 def remove_stopwords(rev):
    # iNPUT : IT WILL TAKE ROW/REVIEW AS AN INPUT
    # take the paragraph, break into words, check if the word is a stop word, remove if stop word, combine the words into a para again
    review_tokenized = word_tokenize(rev)
    rev_new = " ".join([i for i in review_tokenized  if i not in stop_words])
    return rev_new

# Removing stopwords
 df9['Idea_processed'] = [remove_stopwords(r) for r in df9['Idea_processed']]
# Begin Lemmatization 
 nltk.download('wordnet')
 nltk.download('omw-1.4')
 nltk.download('averaged_perceptron_tagger')
 from nltk.stem import WordNetLemmatizer
 from nltk.corpus import wordnet

# function to convert nltk tag to wordnet tag
 lemmatizer = WordNetLemmatizer()

# Finds the part of speech tag
# Convert the detailed POS tag into a shallow information
 def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

# lemmatize sentence using pos tag
 def lemmatize_sentence(sentence):
  # word tokenize -> pos tag (detailed) -> wordnet tag (shallow pos) -> lemmatizer -> root word
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  # output will be a list of tuples -> [(word,detailed_tag)]
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged) # output -> [(word,shallow_tag)]
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


 df9['Idea_processed'] = df9['Idea_processed'].apply(lambda x: lemmatize_sentence(x))



# Importing module
 from sklearn.feature_extraction.text import TfidfVectorizer

# Creating matrix of top 2500 tokens
 tfidf = TfidfVectorizer(max_features=2500)
# tmp_df = tfidf.fit_transform(df.review_processed)
# feature_names = tfidf.get_feature_names()
# pd.DataFrame(tmp_df.toarray(), columns = feature_names).head() 

 X = tfidf.fit_transform(df9.Idea_processed).toarray()
 y = df9['deal/no deal'].map({'Deal' : 1, 'No Deal' : 0}).values
 featureNames = tfidf.get_feature_names_out()





# # Splitting the dataset into train and test
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

 from sklearn.tree import DecisionTreeClassifier

 dt = DecisionTreeClassifier()
 dt.fit(X_train,y_train)

 y_pred = dt.predict(X_test)

 from sklearn.metrics import confusion_matrix, accuracy_score
 accuracy = accuracy_score(y_test, y_pred)

 from sklearn.metrics import roc_auc_score
 ra = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])

 ##featureImportance.sort_values(by='Importance')
 featureImportance = pd.DataFrame({i : j for i,j in zip(dt.feature_importances_,featureNames)}.items(),columns = ['Importance','word'])
 fi=featureImportance.sort_values(by='Importance',ascending=False)
 
 return df9, accuracy, ra, fi


st.set_page_config(layout="wide", initial_sidebar_state='expanded')
# Add back ground image



selected = option_menu(menu_title=None, options=["ML model","Data visualization","Code"], icons=["clipboard-data","award","coin"], orientation="horizontal")

if selected=="ML model":
 def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.ytimg.com/vi/-TO4HaPtVH8/maxresdefault.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

 add_bg_from_url()
 
# GUI.py
 cls1,cls2,cls3=st.columns((0.5,3,0.5))
 with cls2:
  st.title("Will you get :orange[Funding] for your Idea :orange[?]")

 cl1,cl2,cl3=st.columns((1,2,1))
 
 with cl2:
# Get input from user for hashtag
  user_sentence = st.text_input("Enter the idea:")

  if st.button("predict"):  
   b = shark(user_sentence)
   progress_bar = cl2.progress(0)
   for perc_completed in range(100):
    time.sleep(0.05)
    progress_bar.progress(perc_completed+1)
   st.button(b)

if selected=="Data visualization":
 st.markdown('### Metrics')
 col1, col2, col3 = st.columns(3)
 col1.metric('Total pitch', '# 163')
 col2.metric('Deal', '# 95')
 col3.metric('No Deal', '# 68')
 with st.sidebar:
  selected = option_menu(menu_title=None, options=["About Data","About Investor"], icons=["clipboard-data","coin"], orientation="horizontal")
  if selected == "About Investor":
    option1= st.selectbox("Select Investor 1",("Amit Jain","Namita","Anupam","Vineeta","Aman","Piyush"))
    option2= st.selectbox("Select Investor 2",("Namita","Amit Jain","Anupam","Vineeta","Aman","Piyush"))


 if selected == "About Investor":
   if st.sidebar.button("show"):
    st.markdown('''----''')
    c1,c2 = st.columns((6,4))
    with c1:
        c= invest()
        st.markdown('##### Comparing no of deals done by investors')
        fig = px.pie(c, values="Deal", names="index", hole=0.75, width=400, height=300)
        st.write(fig)

    with c2:
      
        df7= interest()
        st.markdown('##### Two investors invested in same company')
        e=[df7["investor"].value_counts()]
        f=pd.DataFrame(e)
        f.reset_index(inplace = True)
        fig2 = px.bar(f, x="index", y=["Deal"], text_auto=True, width=400, height=300)
        st.write(fig2)
    
    st.markdown('''----''')
    st.markdown('##### Ideas where both of them invested')
    with st.expander("click to read"):
        st.dataframe(df7["Idea"])

 if selected == "About Data":
    if st.sidebar.button("show"):
     df9, accuracy, ra, fi=data()
     st.markdown('''----''')
     st.markdown("##### Data frame head()")
     st.dataframe(df9.head())
     st.markdown('''----''')
     st.markdown("##### ML model score")
     colu1,colu2=st.columns(2)
     colu1.metric('Accuracy of ML Model', accuracy)
     colu2.metric('Auroc Score', ra)
     st.markdown('''----''')
     st.markdown("##### Feature impotance")
     with st.expander("click to read"):
        st.dataframe(fi)
if selected=="Code":

   def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.ytimg.com/vi/-TO4HaPtVH8/maxresdefault.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

   add_bg_from_url()
   colu1,colu2,colu3=st.columns((1,2,1))
   with colu2:
    with open('C:/Users/DELL/sharktankindia2.pdf', 'rb') as f:
     base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    st.markdown(f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">', unsafe_allow_html=True)





        
     
 