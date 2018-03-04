import scipy.sparse as sps
import numpy as np
cimport numpy as np

class SLIM_RMSE():
    def __init__(self, URM_train):
        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.URM_mask = self.URM_train.copy()
        #S = sps.csr_matrix((self.n_items, self.n_items), dtype=np.float32)

    #Calcola il prodotto scalare tra la URM[(user_min-1):(user_max), :] e S_matrix
    #Il -1 qui sopra è presente poiché gl user partono da 1 e sono consecutivi, quindi la URM
    #è grande max(users)

    def scalar_product(self, user_min, user_max, S):
        cdef int x = 0
        cdef prod = 0
        user_min = int(user_min)
        user_max = int(user_max)
        support_matrix = np.zeros((user_max - user_min + 1, np.size(S, 1)))

        #Trovare la prima occorrenza di user_min e l'ultima di user_max
        cdef int target_min = 0
        cdef int target_max = 0
        while (int(self.users[target_max]) != user_max):
            if int(self.users[target_min]) != user_min:
                target_min += 1
            if (user_min != user_max):
                target_max += 1
            if (int(self.users[target_max]) == user_max):
                while (target_max < np.size(self.users) - 1 and self.users[target_max + 1] == user_max):
                    target_max += 1

        cdef int starting_point_current_user = target_min
        cdef int starting_point_next_user, j
        for x in range(0, np.size(support_matrix, 0), 1):
            print(x)
            for col_s in range(0, np.size(S, 1), 1):
                j = starting_point_current_user
                prod = 0
                while(j <= target_max):
                    prod += self.ratings[j] * S[int(self.items[j]), col_s]
                    j += 1
                    if (j == np.size(self.users) or int(self.users[j]) != int(self.users[j - 1])):
                        support_matrix[x, col_s] = prod
                        starting_point_next_user = j
                        break
            starting_point_current_user = starting_point_next_user
        return support_matrix

    def SLIM_RMSE_epoch(self, URM_train, users, movies, ratings):
        #cdef  des = np.zeros([n-m+1,m]) per prova per Cython, non funziona
        self.users = users
        self.items = movies
        self.ratings = ratings
        print("Initializing S matrix randomly...")
        #S = np.ones((self.n_items, 2))
        S = np.random.rand(self.n_items, self.n_items)
        cdef float beta = 0.1
        cdef int gamma = 10
        cdef int j = 0
        cdef int i = 0
        #Quella qua sotto è una prova con solo una colonna di S
        product = self.scalar_product(np.min(users), self.n_users, S)
        
        function = 1/2*(np.linalg.norm(URM_train- product,2))**2 + (beta/2)*(np.linalg.norm(S[:,j]))**2 + gamma
        gradient_w_i_j = (URM_train[i,j] - self.scalar_product(users[i], users[i], S))*(-URM_train[i,i]) + beta*S[i,j] + gamma
        print(function, gradient_w_i_j)


