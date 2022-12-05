<<<<<<< HEAD
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

# I. 
def main():
    excel_file_url = r"C:\Code\Brand_details.xlsx"
    tab = "data"
    return excel_file_url, tab 

# Read and modify existing dataframe for further analysis 
def read_excel_and_modify_data():
    excel_file_url, tab = main()
    
    original_data = pd.read_excel(excel_file_url, sheet_name = tab)
    original_data = original_data.iloc[2:,1:].reset_index(drop=True)
    original_data.columns = original_data.iloc[0]
    original_data = original_data[1:]
    
    modified_data = original_data.rename(columns={"Liked/not liked":"Liked"})
    just_liked_brands = modified_data[modified_data.Liked == "yes"]
    
    one_zero_modified_data = modified_data
    one_zero_modified_data["Liked"] = one_zero_modified_data["Liked"].replace(['yes', 'no'], ['1', '0'])
    one_zero_modified_data.Liked = one_zero_modified_data.Liked.astype(float)
    
    return one_zero_modified_data
    
read_excel_and_modify_data()

# Non-personalized recommandation - directed to all users, without taking into consideration users preferences
class non_personalized_recommendation:
    
# Find the brand which is liked the most by users
    def top_brands_for_users():
        just_liked_brands = read_excel_and_modify_data()
        count_likes = just_liked_brands['Brand'].value_counts() 
        count_likes.head(3)
        return count_likes

#check the ratio between - liked/not like e.g. 0.5 => 50% of consumers liked the product     
    def check_likes_ratio():
        one_zero_modified_data = read_excel_and_modify_data()
        likes_ratio = one_zero_modified_data[["Brand", "Sub Brand", "Liked"]].groupby(['Brand','Sub Brand']).mean()
        likes_ratio = round(likes_ratio,2)
        likes_ratio = likes_ratio.sort_values(by="Liked", ascending = False)
        return likes_ratio
    
    def check_how_many_consumers_voted():
        modified_data = read_excel_and_modify_data()
        consumer_number = modified_data["Brand"].value_counts()
        frequently_tasted_brands = consumer_number[consumer_number > 2].index
        return frequently_tasted_brands

# include rows in df that include brands tasted by consumers > 2 times and check in this group the liking ratio    
    def likes_ratio_just_for_often_rated_brands():
        one_zero_modified_data = read_excel_and_modify_data()
        frequently_tasted_brands = non_personalized_recommendation.check_how_many_consumers_voted()
        popular_brands = one_zero_modified_data[one_zero_modified_data["Brand"].isin(frequently_tasted_brands)]
        popular_brands_avg = popular_brands[["Brand","Liked"]].groupby("Brand").mean().sort_values(by="Liked", ascending = False)
        return popular_brands_avg

    
print("Three top brands which users liked the most:\n", non_personalized_recommendation.top_brands_for_users() )
print("Subbrand 'likes ratio' for Pepsico brands:\n", non_personalized_recommendation.check_likes_ratio())
print("Brands for which consumers ranked more that 2 times:\n", non_personalized_recommendation.check_how_many_consumers_voted())
print("'Like' ratio for brands which were ranked more than 2 times:\n", non_personalized_recommendation.likes_ratio_just_for_often_rated_brands().sort_values(by="Liked", ascending = False))

# --> Conclusion - Doritos brand was tasted the most and the brand had the highest 'liking' ratio
# --> Sunbites high ranking was disrupted due to low number of cosumers who tasted the brand   

#II

def create_pairs(x):
    pairs = pd.DataFrame(list(permutations(x.values, 2)),
                              columns=["brand_1","brand_2"])      
    return pairs

#check the "most common" pair of brands to make suggestions 
class make_suggestion_find_most_common_pair_of_brands:
       
 #function to check the "most common" pair of brands
     def modify_data_for_pairs():
         just_liked_brands = read_excel_and_modify_data()
         brands_pair = just_liked_brands.groupby("User")["Brand"].apply(create_pairs).reset_index(drop=True)
         return brands_pair
#count the pairs, how often each combination occurs
     def quantity_of_brand_pair_combination(brand_name):
         brands_pair = make_suggestion_find_most_common_pair_of_brands.modify_data_for_pairs()
         pair_counts = brands_pair.groupby(["brand_1","brand_2"]).size() 
         pair_counts_df = pair_counts.to_frame(name = "size").reset_index().sort_values("size",ascending = False)
         check_pair = pair_counts_df[pair_counts_df["brand_1"] == brand_name]
         return check_pair
print("Check most common brand pairs:\n", make_suggestion_find_most_common_pair_of_brands.quantity_of_brand_pair_combination("LAYS"))    

# --> Conclusion e.g. For consumers who are buying LAYS brand it can be recommended also Doritos (from ealier section we know that Doritos was a brand with relative high liking ratio)

#FOCUS NOW ON IT
### Content-based recommendations ###
class content_based_recommendations: 
    
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

#IV

#Matrix factorization - decompose user rating metrics into product into 2 lower dimention (factors)
#Sparsity check - how many empty values in total 
check_empty = user_data.isnull().values.sum()
check_empty
total = user_data.size
sparsity = check_empty/total
sparsity 
print ("Dataset is completed with", round(sparsity,2)*100 , "%")

#check how many brand are ranked by users
user_data.notnull().sum()
#factors can be found if there is at leat one value in every row and column 
#every consumer gave at least 1 rating & every item has been ranked at least once 

#singular_value_docomposition(SVD) - finds factors for your metrics



 
=======
>>>>>>> 2335994efb6bcb95149e661f6e4d99675bf76d59

