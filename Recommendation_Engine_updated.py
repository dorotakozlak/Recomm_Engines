
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
    
print("This is DataFrame that will be used for further analysis:\n", read_excel_and_modify_data())

# Non-personalized recommandation - directed to all users, without taking into consideration users preferences
class NonPersonalizedRecommendation:
    
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
        frequently_tasted_brands = NonPersonalizedRecommendation.check_how_many_consumers_voted()
        popular_brands = one_zero_modified_data[one_zero_modified_data["Brand"].isin(frequently_tasted_brands)]
        popular_brands_avg = popular_brands[["Brand","Liked"]].groupby("Brand").mean().sort_values(by="Liked", ascending = False)
        return popular_brands_avg

    
print("Three top brands which users liked the most:\n", NonPersonalizedRecommendation.top_brands_for_users() )
print("Subbrand 'likes ratio' for Pepsico brands:\n", NonPersonalizedRecommendation.check_likes_ratio())
print("Brands for which consumers ranked more that 2 times:\n", NonPersonalizedRecommendation.check_how_many_consumers_voted())
print("'Like' ratio for brands which were ranked more than 2 times:\n", NonPersonalizedRecommendation.likes_ratio_just_for_often_rated_brands().sort_values(by="Liked", ascending = False))

# --> Conclusion - Doritos brand was tasted the most and the brand had the highest 'liking' ratio
# --> Sunbites high ranking was disrupted due to low number of cosumers who tasted the brand   

#II

def create_pairs(x):
    pairs = pd.DataFrame(list(permutations(x.values, 2)),
                              columns=["brand_1","brand_2"])      
    return pairs

#check the "most common" pair of brands to make suggestions 
class MakeSuggestionFindMostCommonPairOfBrands:
       
 #function to check the "most common" pair of brands
     def modify_data_for_pairs():
         just_liked_brands = read_excel_and_modify_data()
         brands_pair = just_liked_brands.groupby("User")["Brand"].apply(create_pairs).reset_index(drop=True)
         return brands_pair
#count the pairs, how often each combination occurs
     def quantity_of_brand_pair_combination(brand_name):
         brands_pair = MakeSuggestionFindMostCommonPairOfBrands.modify_data_for_pairs()
         pair_counts = brands_pair.groupby(["brand_1","brand_2"]).size() 
         pair_counts_df = pair_counts.to_frame(name = "size").reset_index().sort_values("size",ascending = False)
         check_pair = pair_counts_df[pair_counts_df["brand_1"] == brand_name]
         return check_pair
print("Check most common brand pairs:\n", MakeSuggestionFindMostCommonPairOfBrands.quantity_of_brand_pair_combination("LAYS"))    

# --> Conclusion e.g. For consumers who are buying LAYS brand it can be recommended also Doritos (from ealier section we know that Doritos was a brand with relative high liking ratio)


### Content-based recommendations ###

#based on the similarities based on items user liked in the past
#jaccard similarity (fron library) - ratio of attributes they have in common divided by total sumber od attributies combined
#below useful to check similarities between single items

class ContentBasedRecommendations: 
    
    def modified_table():
        data_for_content = read_excel_and_modify_data()
        brand_flavours_df = pd.crosstab(data_for_content["Brand"], data_for_content["HARMONIZED_FLAVOUR"])
        return brand_flavours_df
        
    def jaccard_similarity(brand1, brand2):
        brand_flavours_df = ContentBasedRecommendations.modified_table()
        brand1_row = brand_flavours_df.loc[brand1]
        brand2_row = brand_flavours_df.loc[brand2]
        check_jaccard_score = jaccard_score(brand1_row, brand2_row, average = "macro") 
        return check_jaccard_score      

#print("Jaccard similarity between Lays and Doritos is", ContentBasedRecommendations.jaccard_similarity("DORITOS", "LAYS"))
#micro - Calculate metrics globally by counting the total true positives, false negatives and false positives
#macro - Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

#check all similarities; pdist- calculate distance between observations

    def jaccard_similarity_all():
        brand_flavours_df = ContentBasedRecommendations.modified_table()
        jaccard_distances = pdist(brand_flavours_df.values, metric='jaccard')
        square_jaccard_distances = squareform(jaccard_distances)
        jaccard_similarities_array = 1 - square_jaccard_distances
        distance_df = pd.DataFrame(jaccard_similarities_array,##create distance table for all brands that are available 
                                   index = brand_flavours_df.index,
                                   columns = brand_flavours_df.index)
        return distance_df

print("Jaccard similarity between Lays and Doritos is", ContentBasedRecommendations.jaccard_similarity("LAYS","DORITOS"))    
print("Distances for all brands \n", ContentBasedRecommendations.jaccard_similarity_all())
     
#print("Similarities for Doritos brand:\n", distance_df['DORITOS'].sort_values(ascending=False))

# III

##User profile recommendations 
#Find similar users and based on it check items which they liked 

class UserProfileRecommendations:
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
    def cosine_similarity_function(brand_flavour1, brand_flavour2):
        
        brand_data_pivot = UserProfileRecommendations.modified_brand_data()
        cosine_similarity_check = cosine_similarity(brand_data_pivot.loc[brand_flavour1, :].values.reshape(1,-1),
                          brand_data_pivot.loc[brand_flavour2, :].values.reshape(1,-1))
        return cosine_similarity_check 

#similarity metrics between all items 
    def cosine_similarity_all_items(): 
        brand_data_pivot = UserProfileRecommendations.modified_brand_data()
        similarities = cosine_similarity(brand_data_pivot)
        similarities_df = pd.DataFrame(similarities,
                                       index = brand_data_pivot.index,
                                       columns = brand_data_pivot.index)
        return similarities_df
    
#based on this similarity metrics it is possible to create recommendations e.g.
#the most similar brand to Doritos Paprika is Star Salt based on the cunsumer preferences
    def the_most_similar_brand(similar_brand):
        similarities_df = UserProfileRecommendations.cosine_similarity_all_items() 
        cosine_similarity_item = similarities_df.loc[similar_brand]
        ordered_similarities = cosine_similarity_item.sort_values(ascending=False)
        return ordered_similarities 
   

print("Similatrities between all products - \n", UserProfileRecommendations.cosine_similarity_all_items())
print("Similatrity between two products - LAYS Paprika & Star Paprika is \n", UserProfileRecommendations.cosine_similarity_function("LAYS Paprika", "Star Paprika"))
# --> Conclusion - brands seems to be quite similar reg consumers preferences
print("Similatrity between two products - LAYS Paprika & Cheetos Ketchup is \n", UserProfileRecommendations.cosine_similarity_function("LAYS Paprika", "Cheetos Ketchup"))
# --> Conslusion - negative, so brand are quite different from each other
print("The most similar brand to Doritos Paprika is\n", UserProfileRecommendations.the_most_similar_brand("DORITOS Paprika"))


#K-Nearest  neighbors
#how user can feel about item even if not tasted -> user-user similarity 
#here we can see which consumers have similar taste
class KNearestNeighbors:
    def K_nearest_df_similar_users(): 
        user_data_pivot = UserProfileRecommendations.modified_brand_data()
        u_similarities = cosine_similarity(user_data_pivot)
        cosine_similarity_user = pd.DataFrame(u_similarities,
                                              index = user_data_pivot.index,
                                              columns = user_data_pivot.index)
        return cosine_similarity_user

    def find_two_most_similar_users(user1):
        cosine_similarity_user = KNearestNeighbors.K_nearest_df_similar_users()
        #choose here respective column 
        user_similarities_series = cosine_similarity_user.loc[:, user1]
        user_similarities_series = user_similarities_series.sort_values(ascending = False)
        KNN = user_similarities_series[1:3].index # find 2 most similar consumers 
        return KNN
       
    #what rating similar users gave to the product that was not rated by our key consumers
    def check_raitings_of_similar_users(brand1_user):
        KNN = KNearestNeighbors.K_nearest_df_similar_users()
        user_data_pivot = UserProfileRecommendations.modified_brand_data()
        neighbour_data = user_data_pivot.reindex(KNN)
        neighbour_data = neighbour_data[brand1_user].mean() #users similar taste - but if no response then misleasing 
        return neighbour_data

#Scikit-learn KNN method 
    def scikit_learn_KNN_predict_user_rates(brand1_scikit, user1_scikit): 
        user_data_pivot = UserProfileRecommendations.modified_brand_data()
        modified_data_for_user = UserProfileRecommendations.modified_brand_data()
        user_data_pivot2 = user_data_pivot.drop(brand1_scikit, axis=1) #this is target
        target_user_x = user_data_pivot2.loc[[user1_scikit]]
        
        other_users_y = modified_data_for_user[brand1_scikit] #check if 1 or 2 brackets ##original table - how other users liked Star Paprika brand
        other_users_y = other_users_y.dropna(inplace=True)
        other_users_x = user_data_pivot2[other_users_y.notnull()] #with centralized, so we are choosing consumer from orginal table without NaN #we care about consumers who scored the book, so filter just users who tried it
        return target_user_x, other_users_y, other_users_x  #we want to predict user1 

#most likely how User 1 will like "Star Paprika" product
    def KNeighborsRegressor_method(): 
        other_users_y = KNearestNeighbors.scikit_learn_KNN_predict_user_rates("Star Paprika", "USER 1")
        other_users_x = KNearestNeighbors.scikit_learn_KNN_predict_user_rates("Star Paprika", "USER 1")
        target_user_x = KNearestNeighbors.user_data_pivot2.loc[["USER 1"]]
        user_knn = KNeighborsRegressor(metric="cosine", n_neighbors=2)
        user_knn.fit(other_users_x, other_users_y)
        user_user_pred = user_knn.predict(target_user_x)
        return user_user_pred 
    
print("The most similar two users to USER 1 are \n:", KNearestNeighbors.find_two_most_similar_users("USER 1"))
print("Check the taste of similar users for Star Paprika"), KNearestNeighbors.check_raitings_of_similar_users("Star Paprika") 
## -- Conclusion -> most likely User 1 will not like it 
#print("Predict whether how will rate user 1 Star Paprika"), KNearestNeighbors.scikit_learn_KNN_predict_user_rates("Star Paprika", "USER 1")    
#print("\nUser 1 will like Star Paprika:\n", KNearestNeighbors.KNeighborsRegressor_method())



 
