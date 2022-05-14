# File to Find the theta distance between two elements

# import sklearn
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

# Texts - we want to find the distance between
text1 = ["London Paris London","Paris London Paris","London London paris"]
# text2 = "Paris London Paris"

# Used sckitlearn to count the number of words in texts
con_verter = CountVectorizer()

# conversion of text to number of words  
# conversion to distance vector from x and y axis
mat = con_verter.fit_transform(text1)

# checking similarity using cosines 
same_val = cosine_similarity(mat) 

print (same_val)

