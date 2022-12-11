
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
    
    data_for_content = one_zero_modified_data.drop(["User","Sub Brand","TECHNOLOGY","Liked"], axis = 1)
  
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


### Content-based recommendations ###

#based on the similarities based on items user liked in the past
#jaccard similarity (fron library) - ratio of attributes they have in common divided by total sumber od attributies combined
#below useful to check similarities between single items

class content_based_recommendations: 
    def jaccard_similarity(brand1, brand2):
        data_for_content = read_excel_and_modify_data()
        brand_flavours_df = pd.crosstab(data_for_content["Brand"], data_for_content["HARMONIZED_FLAVOUR"])
        brand1_row = brand_flavours_df.loc[brand1]
        brand2_row = brand_flavours_df.loc[brand2]
        check_jaccard_score = jaccard_score(brand1_row, brand2_row, average = "macro") 
        return check_jaccard_score      
#micro - Calculate metrics globally by counting the total true positives, false negatives and false positives
#macro - Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

#check all similarities; pdist- calculate distance between observations
###FOCUS NOW ON IT###
    def jaccard_similarity_all():
        brand_flavours_df = content_based_recommendations.jaccard_similarity()
        jaccard_distances = pdist(brand_flavours_df.values, metric='jaccard')
        square_jaccard_distances = squareform(jaccard_distances)
        jaccard_similarities_array = 1 - square_jaccard_distances
        distance_df = pd.DataFrame(jaccard_similarities_array,##create distance table for all brands that are available 
                                   index = brand_flavours_df.index,
                                   columns = brand_flavours_df.index)
        return distance_df
    
print("Distances for all brands \n", content_based_recommendations.jaccard_similarity_all())
print("Jaccard similarity between Lays and Doritos is", content_based_recommendations.jaccard_similarity("LAYS","DORITOS"))      
#print("Similarities for Doritos brand:\n", distance_df['DORITOS'].sort_values(ascending=False))

# III

##User profile recommendations 
#Find similar users and based on it check items which they liked 
class user_profile_recommendations():
    def modified_brand_data():
        one_zero_modified_data = read_excel_and_modify_data()
        one_zero_modified_data["Brand_Flavour"] = one_zero_modified_data["Brand"] +" " + one_zero_modified_data["HARMONIZED_FLAVOUR"] 
        modified_data_for_user = one_zero_modified_data.drop(one_zero_modified_data.columns[[1,2,3,4]],axis=1)
        modified_data_for_user = modified_data_for_user.pivot(index = "User", 
                                                              columns="Brand_Flavour",
                                                              values ="Liked")
        avg_likes = modified_data_for_user.mean(axis=1) #row means
        user_data_pivot = modified_data_for_user.sub(avg_likes, axis=0) #substract from rest
        user_data_pivot = user_data_pivot.fillna(0)
        brand_data_pivot = user_data_pivot.T
        return brand_data_pivot
#cosine - numpy array - values vary from -1/1 -> 1 is most similar    
    def cosine_similarity(brand_flavour1, brand_flavour2):
        cosine_similarity_check = cosine_similarity(brand_data_pivot.loc[brand_flavour1, :].values.reshape(1,-1),
                          brand_data_pivot.loc[brand_flavour2, :].values.reshape(1,-1))
        return cosine_similarity_check 

#similarity metrics between all items 
    def cosine_similarity_all_items(): 
        brand_data_pivot = user_profile_recommendations.modified_brand_data()
        similarities = cosine_similarity(brand_data_pivot)
        similarities_df = pd.DataFrame(similarities,
                                       index = brand_data_pivot.index,
                                       columns = brand_data_pivot.index)
        return similarities_df
    
#based on this similarity metrics it is possible to create recommendations e.g.
#the most similar brand to Doritos Paprika is Star Salt based on the cunsumer preferences
    def the_most_similar_brand(similar_brand):
        similarities_df = user_profile_recommendations.cosine_similarity_all_items() 
        cosine_similarity_item = similarities_df.loc[similar_brand]
        ordered_similarities = cosine_similarity_item.sort_values(ascending=False)
        return ordered_similarities 
    
print("Similatrities between all products - \n", user_profile_recommendations.cosine_similarity_all_items())

print("Similatrity between two products - LAYS Paprika & Star Paprika is \n", user_profile_recommendations.cosine_similarity("LAYS Paprika", "Star Paprika"))
# --> Conclusion - brands seems to be quite similar reg consumers preferences
print("Similatrity between two products - LAYS Paprika & Cheetos Ketchup is \n", user_profile_recommendations.cosine_similarity("LAYS Paprika", "Cheetos Ketchup"))
# --> Conslusion - negative, so brand are quite different from each other
print("The most similar brand to Doritos Paprika is\n", user_profile_recommendations.the_most_similar_brand("DORITOS Paprika"))


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

#IV ##not finished ##tobedone

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



 
