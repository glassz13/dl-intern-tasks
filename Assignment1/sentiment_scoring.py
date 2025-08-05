import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

# 1. Load the CSV files
comments_df = pd.read_csv("comments.csv")        
engagement_df = pd.read_csv("engagements.csv")     

# 2. Merge on 'post_id'
merged_df = comments_df.merge(engagement_df, on="post_id", how="left")


print("Merged shape:", merged_df.shape)
print(merged_df.head())


# Step 1: Choose the pretrained model name
model_name = "cardiffnlp/twitter-roberta-base-sentiment"

# Step 2: Load tokenizer (converts text to tokens)
token = AutoTokenizer.from_pretrained(model_name)

# Step 3: Load the model itself (RoBERTa fine-tuned on sentiment)
rmodel = AutoModelForSequenceClassification.from_pretrained(model_name) 

# Step 4: Create a pipeline object (easy interface for prediction)
sentiment_model = pipeline("sentiment-analysis", model=rmodel, tokenizer=token)

# Step 5: Apply the model to each comment text
def get_sentiment_label(text):
    result = sentiment_model(text)[0]["label"] 
    if result == "LABEL_0":
        return "negative"
    elif result == "LABEL_1":
        return "neutral"
    elif result == "LABEL_2":
        return "positive"

merged_df["sentiment"] = merged_df["text"].apply(get_sentiment_label)

# Step 6: Preview results
print(merged_df[["text", "sentiment"]].head(10))




# Accuracy
accuracy = accuracy_score(merged_df["true_sentiment"], merged_df["sentiment"])
print(f"Model Accuracy: {accuracy:.2f}")

#classification
print("\nClassification Report:")
print(classification_report(merged_df["true_sentiment"], merged_df["sentiment"]))


sentiment_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

# Apply to each comment
merged_df["sentiment_score"] = merged_df["sentiment"].map(sentiment_map)

# Group all comments by post_id and calculate average sentiment
final_df = merged_df.groupby("post_id").agg({
    "sentiment_score": "mean",              # average sentiment from all comments
    "likes": "first",                       
    "dislikes": "first",
    "avg_read_time_seconds": "first",
    "category": "first"                    
}).reset_index() 



# Normalize each engagement column between 0–1
scaler = MinMaxScaler()
final_df[["likes_norm", "dislikes_norm", "read_time_norm"]] = scaler.fit_transform(
    final_df[["likes", "dislikes", "avg_read_time_seconds"]]
)

# Assign weights to categories based on content value
category_weights = {
    "tutorial": 1.0,       # High value — educates users
    "case-study": 0.8,     # Moderate value — detailed analysis
    "opinion": 0.5         # Low value — subjective content
} 

final_df["category_weight"] = final_df["category"].map(category_weights)

# Compute final relevance score as weighted combination
final_df["relevance_score"] = (
    final_df["sentiment_score"] * 0.4 +                # Sentiment matters most
    final_df["likes_norm"] * 0.2 +                   
    (1 - final_df["dislikes_norm"]) * 0.1 +            # Fewer dislikes = better
    final_df["read_time_norm"] * 0.2 +                 
    final_df["category_weight"] * 0.1                  
) 

print(final_df[["post_id", "category", "sentiment_score", "likes", "dislikes", "avg_read_time_seconds", "relevance_score"]].sort_values(by="relevance_score", ascending=False))

final_df.to_csv("scored_posts.csv", index=False)

print("Saved scored posts to scored_posts.csv")
