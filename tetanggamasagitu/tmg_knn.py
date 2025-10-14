import numpy as np
import pandas as pd
from collections import Counter

class KNeighborsClassifier:
    def __init__(self, k=3):
        self.k = k
        self.x_train = None
        self.y_train = None
        
    def fit(self, x_train, y_train):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        
    def _distance_ecu(self, x_test_point):
        distances = []
        for row in range(len(self.x_train)):

            current_train_point = self.x_train[row]
            total = 0
            
            for col in range(len(current_train_point)):
                total += (current_train_point[col] - x_test_point[col])**2

            distance = np.sqrt(total)
            distances.append(distance)
                
        return pd.DataFrame({
            "distance" : distances
        })
        
    def _get_neighbors(self, distances):
        sorted_distance = distances.sort_values(by='distance', ascending=True)
        nearest = sorted_distance[:self.k]
        return nearest
    
    def _vote(self, nearest):
        neighbor = self.y_train[nearest.index]
        counter = Counter(neighbor)
        prediction = counter.most_common(1)[0][0]
        return prediction
    
    def predict(self,x_test):
        predictions = []
        for test in x_test:
            distances = self._distance_ecu(test)
            nearest = self._get_neighbors(distances)
            pred = self._vote(nearest)
            predictions.append(pred)

        return predictions
    
    