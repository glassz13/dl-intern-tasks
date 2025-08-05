In this assignment, we had to work on blog post comments.

Our tasks included:
- Building a model to analyze comments and label them as  
  "positive", "negative", or "neutral"
- Checking how our model performed
- Defining a formula that gives us a relevance_score for each post

**Note:** We manually created small sample CSV files (`comments.csv` and `engagements.csv`) for this assignment, as no dataset was provided.

---

## The steps we followed are as follows:

1. **Read two CSV files:**  
   `comments.csv` which contains 50 rows, and `engagements.csv` with 10 posts. We merged them using `post_id`.

2. **Selected the model:**  
   We chose the RoBERTa model (`cardiffnlp/twitter-roberta-base-sentiment`) which is pretrained on similar data.  
   We used a tokenizer to convert text into a format understandable by the model.

3. **Built a pipeline** using the tokenizer and model for sentiment analysis.

4. **Label processing:**  
   The model gave output as `LABEL_0`, `LABEL_1`, and `LABEL_2`, so we created a function to map these to  
   `"negative"`, `"neutral"`, and `"positive"` respectively.  
   This was saved in a new column called `sentiment`.  
   We printed sample results showing `text` and its `sentiment`.

5. **Model Evaluation:**  
   We calculated model accuracy (which was `0.76` for our small dataset) and generated a classification report including precision, recall, f1-score, and support.

6. **Mapped sentiment values to scores:**
   - Positive = `1`  
   - Neutral = `0`  
   - Negative = `-1`  
   This was stored in a new column called `sentiment_score`.

7. **Grouped the data by `post_id`** and normalized values like `likes`, `dislikes`, and `read_time`.  
   We also assigned weights to post categories: `tutorial`, `case-study`, and `opinion`.

8. **Final relevance score formula:**

```python
final_df["relevance_score"] = (
    final_df["sentiment_score"] * 0.4 +               
    final_df["likes_norm"] * 0.2 +                    
    (1 - final_df["dislikes_norm"]) * 0.1 +           
    final_df["read_time_norm"] * 0.2 +                
    final_df["category_weight"] * 0.1                 
)
```

9. As you can see, **sentiment score** was given the highest weight, followed by **likes** and **read time**, and then **dislikes** and **category type**.

---

## Result

In the end, we got a file named `scored_posts.csv` with the following columns:

- `post_id`
- `category`
- `sentiment_score`
- `likes`
- `dislikes`
- `avg_read_time_seconds`
- `relevance_score`

The `sentiment_score` comes from the model's prediction based on the text.  
Then we normalized features like `likes`, `dislikes`, and `read time`.  
Finally, using our custom formula, we calculated a `relevance_score` for each post to estimate how useful or engaging it might be for the audience.
