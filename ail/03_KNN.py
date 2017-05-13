
def homework(train_X, train_y, test_X):

    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    
    class KNeighborsClassifier:
        
        def __init__(self, distance_method, n_neighbors):
            
            distance_func_dict = {
                "Euclid": self.euclid_distance,
                "Cosine": self.cosine_similarity,
            }
            self.distance_func = distance_func_dict[distance_method]
            self.n_neighbors = n_neighbors

        
        def _euclid_distance(self, train_X, test_X):

            test_vec = np.expand_dims(test_X, axis=1)
            train_vec = np.expand_dims(train_X, axis=0)
            diff_dim3 = test_vec - train_vec        # (test, 1, 864) - (1, train, 864)
            distance_dim3 = np.sqrt(diff_dim3**2)   # (test, train, 864)
            distance_dim2 = np.sum(distance_dim3, axis=1)#TODO:axis check (test, train)

            topK_indices = np.argsort(distance_dim2, axis=1)[:, :self.n_neighbors]    # (test, k)

            return topK_indices


        def euclid_distance(self, train_X, test_X):

            #test_vec = np.expand_dims(test_X, axis=1)
            #train_vec = np.expand_dims(train_X, axis=0)
            #distances = np.linalg.norm(test_vec - train_X, axis=1)
            #topK_indices = np.argsort(distances, axis=1)[:, ::-1][:, :k]    # (test, k)
            topK_indices = []
            for test_vec in test_X:
                distances = np.linalg.norm(test_vec - train_X, axis=1)
                indices = np.argsort(distances)[:self.n_neighbors]
                topK_indices.append(indices)

            return topK_indices


        def cosine_similarity(self, train_X, test_X):

            normed_train_X = train_X / np.linalg.norm(train_X, axis=1, keepdims=True)
            normed_test_X = test_X / np.linalg.norm(test_X, axis=1, keepdims=True)
            cosine_distances = np.dot(normed_test_X, normed_train_X.T)

            #topK_indices = np.argsort(cosine_distances, axis=1)[:, ::-1][:, :k]    # (test, k)
            topK_indices = []
            for row in cosine_distances:
                indices = np.argsort(row)[::-1][:self.n_neighbors]
                topK_indices.append(indices)

            return topK_indices 


        def validate(self, train_X, train_y, valid_X, valid_y):

            # validationデータに対しての精度を算出
            pred_valid_y = self.predict(train_X, train_y, valid_X)
            score = f1_score(valid_y, pred_valid_y, average="macro")

            return score

        
        def predict(self, train_X, train_y, test_X):
            
            topK_indices = self.distance_func(train_X, test_X)
            topK_labels = train_y[topK_indices]
            #test_labels = np.max([[len(np.where(row==i)[0]) for i in range(10)] for row in topK_labels], axis=1)
            #test_labels = np.array([np.max([np.where(row==i) for i in range(10)]) for row in topK_labels])
            test_labels = [sorted(np.array(np.unique(row, return_counts=True)).T, key=lambda i:i[1], reverse=True)[0][0] for row in topK_labels]

            return np.array(test_labels)


    def get_best_model(models, valid_scores):

        best_index = np.argmax(valid_scores)
        print(best_index)
        return models[best_index]




    def preprocess(train_X, train_y, test_X, scale=False):

        new_train_X, valid_X, new_train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=1234)
        if scale:
            scaler = StandardScaler()
            new_train_X = scaler.fit_transform(new_train_X)
            valid_X = scaler.transform(valid_X)
            test_X = scaler.transform(test_X)

        return new_train_X, new_train_y, valid_X, valid_y, test_X


    params = {
        "n_neighbors": [1, 2, 3, 4, 5],
        #"distance_method": ["Cosine",],
        "distance_method": ["Cosine","Euclid"],
    }
    

    # validationの分割とscaling    
    new_train_X, new_train_y, valid_X, valid_y, test_X = preprocess(train_X, train_y, test_X, scale=False)

    valid_scores = []
    models = []
    for distance_method in params["distance_method"]:
        for n_neighbors in params["n_neighbors"]:
            model = KNeighborsClassifier(distance_method, n_neighbors)
            valid_score = model.validate(new_train_X, new_train_y, valid_X, valid_y)  # このスコアを，単純なvalidationでなくcross_validationの平均スコアにする場合は要拡張
            print(distance_method, n_neighbors, valid_score)
            valid_scores.append(valid_score)
            models.append(model)
    # validation最良のパラメータで，再度全trainで学習する？
    best_model = get_best_model(models, valid_scores)
    pred_y = best_model.predict(train_X, train_y, test_X)

    """
    best_score = 0
    for distance_method in params["distance_method"]:
        distance_func = distance_func_dict[distance_method]
        for n_neighbors in params["n_neighbors"]:
            model = KNeighborsClassifier(distance_func, n_neighbors)
            valid_score = model.validate(new_train_X, new_train_y, valid_X, valid_y)
            if valid_score > best_score:
                best_score = valid_score
                best_model = model
    
    pred_y = best_model.predict(train_X, train_y, test_X)
    """
    return pred_y
