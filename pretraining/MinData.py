import numpy as np
import pandas as pd

import csv

# def load_user_item_matrix_Mind_All(max_user=50000, max_item=33195):
#     """
#     this function loads the user x items matrix from the  movie lens data set.
#     Both input parameter represent a threshold for the maximum user id or maximum item id
#     The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
#     set contains only 50000 users and 1330 items
#     :return: user-item matrix
#     """

#     # Number of unique user_ids: 50000
#     # Number of unique book_ids: 33195
#     # Sparsity of the data: 0.9816
#     df = np.zeros(shape=(max_user, max_item))
#     with open("/export/scratch2/home/manel/RecSys_News/goodbook/ratings_filtered_goodbook.csv", 'r') as f: #subset_Mind_O All_2370_allUsers_KNN_fancy_imputation_Mind_k_30

#         for line in f.readlines():
#             user_id, movie_id, rating, genre = line.split(",")
#             user_id, movie_id, rating, genre = int(user_id), int(movie_id), float(rating), str (genre)
#             if user_id <= max_user and movie_id <= max_item:
#                 df[user_id-1, movie_id-1] = rating

#     return df



def load_user_item_matrix_Mind_TrainingSet_server(max_user=711222, max_item=79547):
    """
    This function loads the user x items matrix from the MIND dataset.
    Both input parameters represent a threshold for the maximum user id or maximum item id.
    :return: user-item matrix
    """
    print("Loading user-item matrix...", flush=True)
    
    df = np.full((max_user, max_item), np.nan)
    
    # Read the CSV file using the csv module
    with open("/export/scratch2/home/manel/RecSys_News/MIND/MINDlarge/train/behaviour_News_interaction_train.csv", 'r') as f:
        reader = csv.reader(f)
        
        # Loop over each row in the CSV file
        for row in reader:
            user_id, movie_id, rating = map(int, row[:3])
            
            # Check if the user_id and movie_id are within the specified limits
            # if user_id <= max_user and movie_id <= max_item:
                # Update the corresponding entry in the NumPy array
            df[user_id - 1, movie_id - 1] = rating
    
    print("User-item matrix loaded successfully.", flush=True)
    return df

def load_user_item_matrix_Mind_TrainingSet(max_user= 1000, max_item= 9368): 
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 50000 users and 1330 items
    :return: user-item matrix
    """
    print ("user-item matrix", flush = True)
    df = np.zeros(shape=(max_user, max_item))
    with open("MIND/mind_version1/MIND_Demo_Version1/train/behaviour_News_interaction_train_1000Users.csv", 'r') as f: 
# /export/scratch2/home/manel/RecSys_News/MIND/MINDlarge/train/behaviour_News_interaction_train.csv
        for line in f.readlines():
            user_id, movie_id = line.split(",")
            user_id, movie_id = int(user_id), int(movie_id)#, float(rating)#, str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = 1
    return df

def load_user_item_matrix_Mind_TrainingSet11(max_user= 1000, max_item= 9368): 
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 50000 users and 1330 items
    :return: user-item matrix
    """
    print ("user-item matrix", flush = True)
    df = np.zeros(shape=(max_user, max_item))
    with open("MIND/mind_version1/MIND_Demo_Version1/train/behaviour_News_interaction_train_1000Users.csv", 'r') as f: 
# /export/scratch2/home/manel/RecSys_News/MIND/MINDlarge/train/behaviour_News_interaction_train.csv
        for line in f.readlines():
            user_id, movie_id, rating = line.split(",")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)#, str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating
    return df

# def load_user_item_matrix_Mind_TrainingSet(max_user= 50000, max_item= 33195):
#     # Dictionaries to map original user_ids and item_ids to consecutive integers
#     user_id_mapping = {}
#     item_id_mapping = {}
#     current_user_index = 0
#     current_item_index = 0
#
#     with open("/export/scratch2/home/manel/RecSys_News/MIND/MIND_demo/behaviour_News_interaction_train.csv", 'r') as f:
#         for line in f.readlines():
#             # Split the line into user_id, movie_id, and rating
#             user_id, movie_id, rating = line.strip().split(",")
#
#             # Check if the user_id is in the mapping, if not, assign a new index
#             if user_id not in user_id_mapping:
#                 user_id_mapping[user_id] = current_user_index
#                 current_user_index += 1
#
#             # Check if the movie_id is in the mapping, if not, assign a new index
#             if movie_id not in item_id_mapping:
#                 item_id_mapping[movie_id] = current_item_index
#                 current_item_index += 1
#
#     # Initialize the user-item matrix with zeros
#     df = np.zeros(shape=(len(user_id_mapping), len(item_id_mapping)))
#
#     # Read the file again to populate the matrix using the mapped indices
#     with open("/export/scratch2/home/manel/RecSys_News/MIND/MIND_demo/behaviour_News_interaction_train.csv", 'r') as f:
#         for line in f.readlines():
#             user_id, movie_id, rating = line.strip().split(",")
#             user_id, movie_id, rating = user_id_mapping[user_id], item_id_mapping[movie_id], float(rating)
#
#             # Check if user_id and movie_id are within bounds
#             if 0 <= user_id < len(user_id_mapping) and 0 <= movie_id < len(item_id_mapping):
#                 df[user_id, movie_id] = rating
#
#     return df

# def load_user_item_matrix_Mind_TrainingSet(max_user= 711222, max_item= 79547):
#     user_id_mapping = {}
#     item_id_mapping = {}
#     reverse_user_id_mapping = {}
#     reverse_item_id_mapping = {}
#     current_user_index = 0
#     current_item_index = 0

#     with open("/export/scratch2/home/manel/RecSys_News/MIND/MINDlarge/train/behaviour_News_interaction_train.csv", 'r') as f:
#         for line in f.readlines():
#             user_id, movie_id, rating = line.strip().split(",")

#             if user_id not in user_id_mapping:
#                 user_id_mapping[user_id] = current_user_index
#                 reverse_user_id_mapping[current_user_index] = user_id
#                 current_user_index += 1

#             if movie_id not in item_id_mapping:
#                 item_id_mapping[movie_id] = current_item_index
#                 reverse_item_id_mapping[current_item_index] = movie_id
#                 current_item_index += 1

#     df = np.zeros(shape=(len(user_id_mapping), len(item_id_mapping)))

#     with open("/export/scratch2/home/manel/RecSys_News/MIND/MINDlarge/train/behaviour_News_interaction_train.csv", 'r') as f:
#         for line in f.readlines():
#             user_id, movie_id, rating = line.strip().split(",")
#             user_id, movie_id, rating = user_id_mapping[user_id], item_id_mapping[movie_id], float(rating)
# 
#             if 0 <= user_id < len(user_id_mapping) and 0 <= movie_id < len(item_id_mapping):
#                 df[user_id, movie_id] = rating

#     return df, reverse_user_id_mapping, reverse_item_id_mapping

# # Example usage
# user_item_matrix = load_user_item_matrix_Mind_TrainingSet()
# print(user_item_matrix)


def load_user_item_matrix_Mind_TestSet(max_user=5000, max_item= 15557): #2370 2835 24676 11927 2835
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 50000 users and 1330 items
    :return: user-item matrix
    """
    df = np.zeros(shape=(max_user, max_item))
    with open("MIND/mind_version1/MIND_Demo_Version1/valid/behaviour_News_interaction_valid.csv", 'r') as f: # Flixster/trainingSet_Mind_1.dat New_Flixster/GB_train.csv

        for line in f.readlines():
            user_id, movie_id = line.split(",")
            user_id, movie_id  = int(user_id), int(movie_id) #, float(rating)#, str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = 1

    return df



def load_user_item_matrix_Mind_Test(max_user=5000, max_item= 15557): # 2370 2008
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 50000 users and 1330 items
    :return: user-item matrix
    """
    df = np.zeros(shape=(max_user, max_item))
    with open("/export/scratch2/home/manel/RecSys_News/goodbook/test_small.csv", 'r') as f:

        for line in f.readlines():
            user_id, movie_id = line.split(",")
            user_id, movie_id  = int(user_id), int(movie_id) #, float(rating)#, str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = 1

    return df

def load_user_item_Mind_Complet(max_user=2370, max_item=2835):# 2370
    df = np.zeros(shape=(max_user, max_item))
    with open(
            "Flixster/With_Fancy_KNN/TrainingSet_2370_allUsers_KNN_fancy_imputation_Mind_k_30.dat",
            'r') as f: 
        for line in f.readlines():
            user_id, movie_id = line.split(",")
            user_id, movie_id  = int(user_id), int(movie_id) #, float(rating)#, str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = 1

    return df
"""
def load_user_item_matrix_Mind_limited_ratings(limit=20):
    
    user_item = load_user_item_matrix_Mind_All()
    user_item_limited = np.zeros(shape=user_item.shape)
    for user_index, user in enumerate(user_item):
        # filter rating indices
        rating_index = np.argwhere(user > 0).reshape(1, -1)[0]
        # shuffle them
        np.random.shuffle(rating_index)
        for i in rating_index[:limit]:
            user_item_limited[user_index, i] = user[i]
    #print(np.sum(user_item_limited, axis=1))
    return user_item_limited
"""
def load_user_item_matrix_Mind_trainMasked(max_user=2370, max_item=2835, file_index=-1):
    df = np.zeros(shape=(max_user, max_item))
    masked_files = [
        # ,#0
        
    ]
    with open(masked_files[file_index], 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df




def load_gender_vector_GB(max_user=2370 ): #2370 2008
    """
        this function loads and returns the gender for all users with an id smaller than max_user
        :param max_user: the highest user id to be retrieved
        :return: the gender vector
        """
    gender_vec = []
    with open("Flixster/subset_Mind_User_O.csv", 'r') as f: 
        for line in f.readlines()[:max_user]: 
            user_id, gender, _ = line.split(",") #, location, _, _, _ , _

            if gender == "M":
                gender_vec.append(0)
            else:
                gender_vec.append(1)

    return np.asarray(gender_vec)

def load_user_item_matrix_Mind_masked(max_user=2370, max_item=2835, file_index=-1):
    files = [
        # Here add path to your files. Please note that we start from #0 like in the example 
        "Flixster/BlurMe/All_Mind_blurme_obfuscated_0.01_greedy_avg_top-1.dat",#0
        
    ]
    df = np.zeros(shape=(max_user, max_item))

    with open(files[file_index], 'r') as f:
        for line in f.readlines():
            user_id, movie_id = line.split(",")
            user_id, movie_id  = int(user_id), int(movie_id) #, float(rating)#, str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = 1

    return df
