import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

# Load input files
posts_df = pd.read_csv("scored_posts.csv")
users_df = pd.read_csv("users.csv")

# Function to calculate personalized score
def calculate_score(post, user_sent, user_time):
    sent_diff = abs(post["sentiment_score"] - user_sent)
    time_diff = abs(post["avg_read_time_seconds"] - user_time)
    return post["relevance_score"] - (0.3 * sent_diff) - (0.2 * (time_diff / 300))

# Recommend top 5 posts (max 3 per category) for a user
def get_recommendations(user_row):
    preferred = user_row["preferred_categories"].split(",")
    user_sent = user_row["historical_avg_sentiment"]
    user_time = user_row["avg_engagement_time"]

    filtered = posts_df[posts_df["category"].isin(preferred)].copy()
    scored_posts = []

    for _, post in filtered.iterrows():
        post_data = post.copy()
        post_data["personalized_score"] = calculate_score(post, user_sent, user_time)
        scored_posts.append(post_data)

    scored_posts.sort(key=lambda x: x["personalized_score"], reverse=True)

    final = []
    cat_count = {}
    for post in scored_posts:
        cat = post["category"]
        if cat_count.get(cat, 0) < 3:
            final.append(post)
            cat_count[cat] = cat_count.get(cat, 0) + 1
        if len(final) == 5:
            break

    return final

# Generate recommendations for all users
all_results = []
for i in range(len(users_df)):
    user = users_df.loc[i]
    recs = get_recommendations(user)
    for post in recs:
        post["user_id"] = user["user_id"]
        all_results.append(post)

result_df = pd.DataFrame(all_results)

# Simulate random user clicks (30% chance)
np.random.seed(42)
result_df["clicked"] = np.random.choice([0, 1], size=len(result_df), p=[0.7, 0.3])

# Calculate Precision@5
def calculate_precision(df):
    scores = []
    for uid in df["user_id"].unique():
        user_df = df[df["user_id"] == uid]
        scores.append(user_df["clicked"].mean())
    return np.mean(scores)

# Calculate NDCG@5
def calculate_ndcg(df):
    scores = []
    for uid in df["user_id"].unique():
        user_df = df[df["user_id"] == uid]
        if len(user_df) > 1:
            y_true = [user_df["clicked"].values]
            y_score = [user_df["personalized_score"].values]
            scores.append(ndcg_score(y_true, y_score))
    return np.mean(scores)

# Run evaluation
precision = calculate_precision(result_df)
ndcg = calculate_ndcg(result_df)

# Output evaluation results
print("\nEvaluation Results:")
print(f"Precision@5: {precision:.3f}")
print(f"NDCG@5:     {ndcg:.3f}")

# Save results
result_df.to_csv("personalized_recommendations.csv", index=False)
with open("evaluation_results.txt", "w") as f:
    f.write(f"Precision@5: {precision:.3f}\nNDCG@5: {ndcg:.3f}")
