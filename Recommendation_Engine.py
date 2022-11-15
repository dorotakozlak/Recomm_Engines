#recommendation creation for customers & theit taste reg Pepsico products
import pandas as pd
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import pdist, squareform

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

print("check on github functionalities")
#############

##User profile recommendations 
## add here based on "User profile recommendations"

#############

 