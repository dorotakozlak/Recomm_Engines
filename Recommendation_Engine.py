#recommendation creation for customers & theit taste reg Pepsico products
import pandas as pd
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsRegressor 
from itertools import permutations 
def create_pairs(x):
    pairs = pd.DataFrame(list(permutations(x.values, 2)),#books that are read together
                             columns=["brand_1","brand_2"])      
    return pairs


#################

#Read and modify existing dataframe 
brand_data = pd.read_excel('Brand_details.xlsx', 'data')
data = brand_data.iloc[2:,1:]
data = data.reset_index(drop = True)
data.columns = data.iloc[0]
data = data[1:]
data = data.rename(columns={"Liked/not liked":"Liked"})


### Non-personalized recommandation ###
#Find the brand which is liked the most by users
Fdata = data[data.Liked == "yes"]
count_likes=Fdata['Brand'].value_counts()
print("Brands which users liked the most:\n", count_likes.index)

#check the ratio between - liked/not like e.g. 0.5 => 50% of consumers liked the product 
data["Liked"] = data['Liked'].replace(['yes', 'no'], ['1', '0']) 
data.Liked = data.Liked.astype(float)
avg_likes = data[["Brand", "Sub Brand", "Liked"]].groupby(['Brand','Sub Brand']).mean()
avg_likes = round(avg_likes,2)
sorted_avg = avg_likes.sort_values(by="Liked", ascending = False)
print(sorted_avg)

#check not how many users voted for a single product 
brand_frequency = data["Brand"].value_counts()
print(brand_frequency)
frequently_tasted_brands = brand_frequency[brand_frequency > 2].index
print("Brands rated more than 2 times:\n", frequently_tasted_brands)

#####################################

####CHECK BELOW CODE - "NON-PERSONALIZED RECOMMNEDATIONS" - END OF VIDEO DATACAMP
#final_brands = data[data["Brand"].isin(frequently_tasted_brands)]
#final_brands_avg = frequently_tasted_brands[["Brand","Liked"]].groupby("Brand")
#print(final_brands_avg.heaad())

# --> Conclusion - Doritos brand was tasted the most and the brand had the highest 'liking' ratio
# --> Sunbites high ranking was disrupted due to low number of  who tasted the brand   

### Non-personalized recommandation ###
#make suggestions of finding most common pair of brands 

###### CHECK it as well!!! ####
#create the pair function

#brands_pair = data.groupby("User")["Brand"].apply(create_pairs)
#brands_pair.reset_index(drop=True)
##count the pairs, how often each combination occurs 
#pair_counts = brands_pair.groupby(["brand_1","brand_2"]).size 
#pair_counts_df = pair_counts.to_frame(name = 'size').reset_index()
#pair_conts_sorted = pair_counts_df.sort_values('size', ascending = False)
#print(pair_counts_df.head())

#######################################

### Content-based recommendations ###
#based on the similarities based on items user liked in the past
cont_data = data.drop(["User","Sub Brand","TECHNOLOGY","Liked"], axis = 1)
brand_flavours_df = pd.crosstab(cont_data["Brand"], cont_data["HARMONIZED_FLAVOUR"])
print(brand_flavours_df)
#jaccard similarity
#ratio of attributes they have in common divided by total sumber od attributies combined

#take from jaccard_score library 
#below useful to check similarities between single items
doritos_row = brand_flavours_df.loc["DORITOS"]
lays_row = brand_flavours_df.loc["LAYS"]
#micro - Calculate metrics globally by counting the total true positives, false negatives and false positives
#macro - Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
print("Jaccard similarity between Lays and Doritos is", jaccard_score(doritos_row, lays_row, average = 'macro'))

#check all similarities; pdist- help doing it
jaccard_distances = pdist(brand_flavours_df.values, metric='jaccard')
square_jaccard_distances = squareform(jaccard_distances)

jaccard_similarities_array = 1 - square_jaccard_distances
print(jaccard_similarities_array)



##create distance table for all brands that are available 
distance_df = pd.DataFrame(jaccard_similarities_array,
                           index = brand_flavours_df.index,
                           columns = brand_flavours_df.index)
print(distance_df)
print(distance_df['LAYS']['Star'])
print(distance_df['DORITOS'].sort_values(ascending=False))


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
#cosine - numpy array 
cosine_similarity(brand_data_pivot.loc["LAYS Paprika", :].values.reshape(1,-1),
                  brand_data_pivot.loc["Star Paprika", :].values.reshape(1,-1))
#above - two brands quite similar
cosine_similarity(brand_data_pivot.loc["LAYS Paprika", :].values.reshape(1,-1),
                  brand_data_pivot.loc["Cheetos Cheese", :].values.reshape(1,-1))
#above - value negative so not similar 

#the most similar items
#CHECK IT IF SIMILARITIES ADDED PROPERLY #STH WRONG HERE!! Why NA?
similarities = cosine_similarity(brand_data_pivot)
cosine_similarity_df = pd.DataFrame(brand_data_pivot,#THIS!!!
                                    index = brand_data_pivot.index,
                                    columns = brand_data_pivot.index)

cosine_similarity_item = cosine_similarity_df.loc["LAYS Salt"]
ordered_similarities = cosine_similarity_item.sort_values(ascending=False)
print(ordered_similarities)


#K-Nearest  neighbors
#how user can feel about item even if not tasted -> user-user similarity 
#CHECK THIS SIMILARITIES!!!
similarities = cosine_similarity(user_data_pivot)
cosine_similarity_user = pd.DataFrame(similarities, #CHECK_IT - Again no data!!! 
                                      index = user_data_pivot.index,
                                      columns = user_data_pivot.index)
cosine_similarity_user.head()

user_similarities_series = cosine_similarity_user.loc["USER 1"]
ordered_similarities = user_similarities_series.sort_values(ascending = False)
KNN = ordered_similarities[1:3].index

print(KNN)

neighbour = user_data_pivot.reindex(KNN)
neighbour["Star Salt"].mean() #users similat to them gave it - but if no response then misleasing 

#SCIKIT-LEARN KNN -SOMETHING WRONG HERE!!!
user_data_pivot.drop("Star Salt", axis=1, inplace=True)
target_user_x = user_data_pivot.loc[["USER 1"]]
print(target_user_x)

other_users_y = user_data["Star Salt"]
other_users_x = user_data_pivot[other_users_y.notnull()]
print(other_users_x)

other_users_y.dropna(inplace=True)
print(other_users_y) #data you want to predict


user_knn = KNeighborsRegressor(metric="cosine", n_neighbors=2)
user_knn.fit(other_users_x, other_users_y)
user_user_pred = user_knn.predict(target_user_x)
print(user_user_pred)
