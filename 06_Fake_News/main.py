import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def main():
    df=pd.read_csv("news.csv").dropna(subset=["text","label"])
    Xtr,Xte,ytr,yte=train_test_split(df.text,df.label,test_size=.2,random_state=42,stratify=df.label)
    model=Pipeline([("tfidf",TfidfVectorizer(stop_words="english",ngram_range=(1,2),max_features=30000)),("clf",LogisticRegression(max_iter=1000))])
    model.fit(Xtr,ytr)
    print(classification_report(yte,model.predict(Xte)))
    while True:
        s=input("News text (q to quit): ").strip()
        if s.lower()=="q": break
        if s: print("Predicted dataset label:",model.predict([s])[0])
if __name__=="__main__": main()
