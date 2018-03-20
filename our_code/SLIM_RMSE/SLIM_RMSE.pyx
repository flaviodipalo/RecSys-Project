import scipy.sparse as sps
import numpy as np
cimport numpy as np
import random

cdef class SLIM_RMSE:
    cdef int n_users
    cdef int n_items
    cdef double[:] users
    cdef double[:] items
    cdef double[:] ratings

    def __init__(self, URM_train):
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        #S = sps.csr_matrix((self.n_items, self.n_items), dtype=np.float32)

    #Calcola il prodotto scalare tra la URM[(user_min-1):(user_max), :] e S_matrix
    #Il -1 qui sopra è presente poiché gl user partono da 1 e sono consecutivi, quindi la URM
    #è grande max(users)

    def scalar_product(self, user_min, user_max, S, user_ratings_number):
        cdef int prod = 0
        for x in range(0, 1, 1):
            for j in range(0, int(user_ratings_number[user_max + 1] - user_ratings_number[user_min]), 1):
                prod += self.ratings[j] * S[int(self.items[j])]
        return prod

    def random_matrix_S(self, rows, cols):
        cdef double[:,:] S = np.zeros((rows, cols))
        cdef i
        cdef j
        for i in range(0, rows, 1):
            for j in range(0, cols, 1):
                S[i, j] = random.uniform(0, 1)

        return S

    def find_occurrences(self, list, c):
        if c == "u":
            occurrences = np.zeros((self.n_users + 1, 1))
        elif c == "i":
            occurrences = np.zeros((self.n_items, 1))
        x = 0
        for i in range(0, np.size(list), 1):
            if i == 0:
                occurrences[x] = i
                x += 1
            else:
                if list[i] != list[i-1]:
                    x += int(list[i]) - int(list[i-1]) - 1    #needed because items have not totally continuos IDs
                    occurrences[x] = i
                    x += 1
        occurrences[x] = np.size(list)
        return occurrences

    def create_URM_prova(self, users_by_item, ratings_by_item, items_by_items_ratings_number):
        URM_prova = np.zeros((self.n_users, 1))
        for i in range(0, int(items_by_items_ratings_number[1]), 1):            #create only the first column of URM
            URM_prova[int(users_by_item[i]) - 1] = ratings_by_item[i]
            print("USER:", users_by_item[i])
        return URM_prova

    def SLIM_RMSE_epoch(self, URM_train, users, movies, ratings, users_by_item, items_by_item, ratings_by_item):
        self.users = users
        self.items = movies
        self.ratings = ratings
        print("Initializing S matrix randomly...")
        S = np.random.rand(self.n_items, self.n_items)
        print("S", np.shape(S))
        users__by_user_ratings_number = self.find_occurrences(users, "u")
        items_by_items_ratings_number = self.find_occurrences(items_by_item, "i")
        #cdef double[:, :] S = self.random_matrix_S(self.n_items, 1)    non funziona
        cdef float beta = 0.1
        cdef int gamma = 10
        cdef int j = 0
        cdef int i = 0
        URM_prova = self.create_URM_prova(users_by_item, ratings_by_item, items_by_items_ratings_number)         #only the first column of the URM

        gradient = np.zeros((np.size(URM_prova, 0), 1)) #gradient = np.zeros((np.size(URM_train[:, 0], 0), 1))
        for i in range(0, 10):
            print ("Iteration: ", i)
            product = self.scalar_product(int(np.min(users)) - 1, int(self.n_users) - 1, S[:, 0], users__by_user_ratings_number)
            function = 1/2*(np.linalg.norm(URM_prova - product, "fro"))**2 + (beta/2)*(np.linalg.norm(S[:,j]))**2 + gamma
            for x in range(0, self.n_users, 1):
                partial = (URM_prova[x] - self.scalar_product(x, x, S[:, 0], users__by_user_ratings_number)) # prova = (URM_train[x,0] - self.scalar_product(x, x, S[:, 0], users__by_user_ratings_number))
                gradient[x] = partial * (-URM_prova[x])# + beta*S[x,0] #gradient[x] = prova*(-URM_train[x, 0])# + beta*S[x,0]
            URM_prova = URM_prova - 0.2*gradient #URM_train[:, 0] = URM_train[:, 0] - 0.2*gradient
            print(function)