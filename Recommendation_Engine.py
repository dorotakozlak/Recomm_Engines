# Checking customers taste regarding Pepsico Snacks in order to provide further product recommendation 
# For this purpose very simple dataset was created  

# import required libraries 
import pandas as pd
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.neighbors import KNeighborsClassifier
from itertools import permutations 

# create function to check the "most common" pair of brands
def create_pairs(x):
    pairs = pd.DataFrame(list(permutations(x.values, 2)),
                             columns=["brand_1","brand_2"])      
    return pairs

###

# I. 

# Read and modify existing dataframe for further analysis 
brand_data = pd.read_excel('Brand_details.xlsx', 'data')
data = brand_data.iloc[2:,1:]
data = data.reset_index(drop = True)
data.columns = data.iloc[0]
data = data[1:]
data = data.rename(columns={"Liked/not liked":"Liked"})
print("data")

# Non-personalized recommandation - directed to all users, without taking into consideration users preferences
# Find the brand which is liked the most by users
Fdata = data[data.Liked == "yes"]
count_likes=Fdata['Brand'].value_counts()
count_likes = count_likes.head(3)
print("Three top brands which users liked the most:\n", count_likes.index)

#check the ratio between - liked/not like e.g. 0.5 => 50% of consumers liked the product 
data["Liked"] = data['Liked'].replace(['yes', 'no'], ['1', '0']) 
data.Liked = data.Liked.astype(float)
avg_likes = data[["Brand", "Sub Brand", "Liked"]].groupby(['Brand','Sub Brand']).mean()
avg_likes = round(avg_likes,2)
sorted_avg = avg_likes.sort_values(by="Liked", ascending = False)
print(sorted_avg)

# check if results are not disrupted - e.g. Sunbites 100% liking, but just 1 conusumer ranked 
brand_frequency = data["Brand"].value_counts()
print(brand_frequency)
frequently_tasted_brands = brand_frequency[brand_frequency > 2].index
print("Brands rated more than 2 times:\n", frequently_tasted_brands)

# include rows in df that include brands tasted by consumers > 2 times and check in this group the liking ratio
final_brands = data[data["Brand"].isin(frequently_tasted_brands)]
final_brands_avg = final_brands[["Brand","Liked"]].groupby("Brand").mean()
print(final_brands_avg.sort_values(by="Liked", ascending = False))

# --> Conclusion - Doritos brand was tasted the most and the brand had the highest 'liking' ratio
# --> Sunbites high ranking was disrupted due to low number of cosumers who tasted the brand   

# II

# make suggestions of finding most common pair of brands 
 
brands_pair = Fdata.groupby("User")["Brand"].apply(create_pairs)
brands_pair = brands_pair.reset_index(drop=True)
print(brands_pair)

#count the pairs, how often each combination occurs 
pair_counts = brands_pair.groupby(["brand_1","brand_2"]).size() 
pair_counts
pair_counts_df = pair_counts.to_frame(name = 'size').reset_index() #to clean up index 
pair_counts_sorted = pair_counts_df.sort_values('size', ascending = False)
check_pair = pair_counts_sorted[pair_counts_sorted["brand_1"] == "LAYS"]
print(check_pair)

# --> Conclusion e.g. For consumers who are buying LAYS brand it can be recommended also Doritos (from ealier section we know that Doritos was a brand with relative high liking ratio)

### Content-based recommendations ###
#based on the similarities based on items user liked in the past
cont_data = data.drop(["User","Sub Brand","TECHNOLOGY","Liked"], axis = 1)
brand_flavours_df = pd.crosstab(cont_data["Brand"], cont_data["HARMONIZED_FLAVOUR"])
print(brand_flavours_df)

#jaccard similarity (fron library) - ratio of attributes they have in common divided by total sumber od attributies combined
#below useful to check similarities between single items
doritos_row = brand_flavours_df.loc["DORITOS"]
lays_row = brand_flavours_df.loc["LAYS"]
#micro - Calculate metrics globally by counting the total true positives, false negatives and false positives
#macro - Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
print("Jaccard similarity between Lays and Doritos is", jaccard_score(doritos_row, lays_row, average = 'macro'))

#check all similarities; pdist- calculate distance between observations
jaccard_distances = pdist(brand_flavours_df.values, metric='jaccard')
square_jaccard_distances = squareform(jaccard_distances)
jaccard_similarities_array = 1 - square_jaccard_distances
print(jaccard_similarities_array)

##create distance table for all brands that are available 
distance_df = pd.DataFrame(jaccard_similarities_array,
                           index = brand_flavours_df.index,
                           columns = brand_flavours_df.index)
print("Distances for all brands \n", distance_df)
print("Similarity for Lays and Cheetos brands:\n", distance_df['LAYS']['Cheetos'])
print("Similarities for Doritos brand:\n", distance_df['DORITOS'].sort_values(ascending=False))

# III

##User profile recommendations 
#Find similar users and based on it check items which they liked 
data["Brand_Flavour"] = data["Brand"] +" " + data["HARMONIZED_FLAVOUR"]
user_data = data.drop(data.columns[[1,2,3,4]],axis=1)
user_data = user_data.pivot(index = "User", 
                                  columns="Brand_Flavour",
                                  values ="Liked")

#NaN - filling with"0" might be misleading, so better to center each value around 0 
avg_likes = user_data.mean(axis=1) #row means
user_data_pivot = user_data.sub(avg_likes, axis=0) #substract from rest
user_data_pivot = user_data_pivot.fillna(0)
brand_data_pivot = user_data_pivot.T
print(brand_data_pivot)

#cosine - numpy array - values vary from -1/1 -> 1 is most similar
cosine_similarity(brand_data_pivot.loc["LAYS Paprika", :].values.reshape(1,-1),
                  brand_data_pivot.loc["Star Paprika", :].values.reshape(1,-1))
# --> Conclusion - brands seems to be quite similar reg consumers preferences

#above - two brands quite similar
cosine_similarity(brand_data_pivot.loc["LAYS Paprika", :].values.reshape(1,-1),
                  brand_data_pivot.loc["Cheetos Ketchup", :].values.reshape(1,-1))
# --> Conslusion - negative, so brand are quite different from each other

#similarity metrics between all items 
similarities = cosine_similarity(brand_data_pivot)
cosine_similarity_df = pd.DataFrame(similarities,
                                    index = brand_data_pivot.index,
                                    columns = brand_data_pivot.index)
cosine_similarity_df

#based on this similarity metrics it is possible to create recommendations e.g.
#the most similar brand to Doritos Paprika is Star Salt based on the cunsumer preferences
cosine_similarity_item = cosine_similarity_df.loc["DORITOS Paprika"]
ordered_similarities = cosine_similarity_item.sort_values(ascending=False)
print(ordered_similarities)


#K-Nearest  neighbors
#how user can feel about item even if not tasted -> user-user similarity 
u_similarities = cosine_similarity(user_data_pivot)
cosine_similarity_user = pd.DataFrame(u_similarities,
                                      index = user_data_pivot.index,
                                      columns = user_data_pivot.index)
cosine_similarity_user #here we can see which consumers have similar taste 

user_similarities_series = cosine_similarity_user.loc["USER 1"]
ordered_similarities = user_similarities_series.sort_values(ascending = False)
KNN = ordered_similarities[1:3].index # find 2 most similar consumers 
KNN

#what rating similar users gave to the product that was not rated by our key consumers
neighbour_data = user_data_pivot.reindex(KNN)
neighbour_data["Star Paprika"].mean() #users similar taste - but if no response then misleasing 
## -- Conclusion -> most likely User 1 will not like it 

#Scikit-learn KNN method 
user_data_pivot2 =user_data_pivot.drop("Star Paprika", axis=1) #this is target
target_user_x = user_data_pivot2.loc[["USER 1"]]
print(target_user_x) #we want to predict USER 1, so seperate it 

#original table - how other users liked Star Paprika brand
other_users_y = user_data["Star Paprika"] #with
print(other_users_y)

#we care about consumers who scored the book, so filter just users who tried it
#with centralized, so we are choosing consumer from orginal table without NaN
other_users_x = user_data_pivot2[other_users_y.notnull()] 
print(other_users_x)


other_users_y.dropna(inplace=True)
print(other_users_y) #data you want to predict

#most likely how User 1 will like "Star Paprika" product
user_knn = KNeighborsRegressor(metric="cosine", n_neighbors=2)
user_knn.fit(other_users_x, other_users_y)
user_user_pred = user_knn.predict(target_user_x)
print("\nUser 1 will like Star Paprika:\n", user_user_pred)

#Classifier method - often used for non-numeric predcitions -right/wrong 
user_knn = KNeighborsClassifier(metric="cosine", n_neighbors=2)
user_knn.fit(other_users_x, other_users_y)
user_user_pred = user_knn.predict(target_user_x)
print("\nMost probably User 1 will classify brand as :\n", user_user_pred)


#Matrix factorization to be implemented next days!!