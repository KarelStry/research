# Research Vertical Repository

Models
1. Simple counting
2. Simple count + tf*idf
3. Simple counting with "risk" or something else without tf*idf
4. (3) with tf*df weighting
5. Sentiment +- 10 word parameter



Current:
- sentiment, risk, hedging

How to save?
{
    total count: % positive, % neutral, % negative, and associated scores for all of them.
    (7 numbers total per transcript)
}

What's next?
- model 3, but look inside the strings in the dict for risk words. If it doesn't contain a risk word, then we will get rid of it.
- Every topic has a different associated list of risk words
- does individual tfidf weighting vs. overall tfidf weighting make a difference? 

Firm name, Firm Headquarter Address, Date of Earnings call, Summary numbers

