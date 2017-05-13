def homework(train_X, train_y, test_X):

    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    
    class KNeighborsClassifier:
        
        def __init__(self, n_neighbors):
            
            self.n_neighbors = n_neighbors


        def validate(self, train_X, train_y, valid_X, valid_y):

            pred_valid_y = self.predict(train_X, train_y, valid_X)
            score = f1_score(valid_y, pred_valid_y, average="macro")

            return score


        def predict(self, train_X, train_y, test_X):

            normed_train_X = train_X / np.linalg.norm(train_X, axis=1, keepdims=True)
            pred_y = []
            for row in test_X:
                normed_row = row / np.linalg.norm(row)
                distances = np.dot(normed_row, normed_train_X.T)
                topK_indices = np.argsort(distances)[::-1][:self.n_neighbors]
                topK_labels = train_y[topK_indices]
                test_labels = sorted(np.array(np.unique(topK_labels, return_counts=True)).T, key=lambda i:i[1], reverse=True)[0][0]
                pred_y.append(test_labels)

            return pred_y


    def get_best_model(models, valid_scores):

        best_index = np.argmax(valid_scores)
        return models[best_index]


    params = {
        "n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    

    new_train_X, valid_X, new_train_y, valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=1234)

    valid_scores = []
    models = []
    for n_neighbors in params["n_neighbors"]:
        model = KNeighborsClassifier(n_neighbors)
        valid_score = model.validate(new_train_X, new_train_y, valid_X, valid_y)
        valid_scores.append(valid_score)
        models.append(model)

    best_model = get_best_model(models, valid_scores)
    pred_y = best_model.predict(train_X, train_y, test_X)


    return pred_y
