import argparse, re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TRAIN_TEXT = [
 "charged twice incorrect bill payment deducted money","refund not received refund delayed",
 "payment failed money deducted billing issue","delivery late package delayed tracking inaccurate",
 "tracking status wrong shipment delayed","app crashes checkout error cannot place order",
 "website confusing cannot find option","support agent helpful issue resolved quickly",
 "friendly representative quick resolution excellent service","long wait no response complaint ignored",
 "support did not answer agent unavailable","great product easy checkout smooth experience",
 "excellent quality on time delivery","simple payment and user friendly app"
]
TRAIN_LABEL = ["negative","negative","negative","negative","negative","negative","negative",
 "positive","positive","negative","negative","positive","positive","positive"]

def clean(text):
    return re.sub(r"[^a-z0-9\\s]", " ", str(text).lower()).strip()

def theme(text):
    t=clean(text)
    rules={"Billing & Payments":["bill","charg","payment","deduct"],
           "Refunds":["refund","return"],
           "Delivery & Tracking":["deliver","late","track","shipment","package"],
           "App & Checkout":["app","crash","checkout","website"],
           "Customer Support":["support","agent","response","wait","representative"]}
    scores={k:sum(w in t for w in words) for k,words in rules.items()}
    best=max(scores,key=scores.get)
    return best if scores[best] else "Other"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="data/customer_feedback.csv")
    ap.add_argument("--output",default="outputs")
    args=ap.parse_args()
    df=pd.read_csv(args.input)
    model=Pipeline([("tfidf",TfidfVectorizer(preprocessor=clean,ngram_range=(1,2))),
                    ("clf",LogisticRegression(max_iter=1000))])
    model.fit(TRAIN_TEXT,TRAIN_LABEL)
    df["sentiment"]=model.predict(df["feedback_text"].fillna(""))
    df["theme"]=df["feedback_text"].apply(theme)
    df["priority"]=df["sentiment"].map(lambda x:"High" if x=="negative" else "Monitor")
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/"feedback_classified.csv",index=False)
    df.groupby(["theme","sentiment"]).size().unstack(fill_value=0).to_csv(out/"theme_sentiment_summary.csv")
    (df[df.sentiment=="negative"].groupby("theme").size()
       .rename("negative_feedback_count").sort_values(ascending=False)
       .to_csv(out/"priority_pain_points.csv"))
    print(df[["feedback_id","sentiment","theme","priority"]].to_string(index=False))
    print("Outputs:",out.resolve())

if __name__=="__main__": main()
