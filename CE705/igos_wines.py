import numpy
import statistics   # Comes with python by default
import random   # Comes with python by default

# Set global values for beta and weights(Randomly generated array with 1 row and m columns whose summation == 1)
# This global values are used in "get_new_weights" function which is used to weight_values that are used in Euclidean formular
# They are passed because the "get_new_weights" function are only allowed 3 parameters (data_matrix, centroids, S) But these values are needed in the calculation.
betaa = 0
weightts = 0

class matrix():
    def __init__(self, csv_file):
        array_2d = numpy.array([])
        self.csv_file = csv_file
        row_from_standardized_matrix = numpy.array([])  # Row from data matrix that will be used in get_distance(Euclidean distance calculation)

    def load_from_csv(self):
        data_csv_contents = open(self.csv_file, 'r').readlines()    # Read the contents of the file line by line into a list

        # Algorithm to store the file contents into array_2d
        counter = 0
        for line in data_csv_contents:
            if counter == 0:
                array_string_format = line.strip('\n').split(',')
                array_float_format = []
                for no in range(0, len(array_string_format)):
                    array_float_format.append(float(array_string_format[no]))
                self.array_2d = numpy.array([array_float_format])
                counter += 1
                continue

            array_string_format = line.strip('\n').split(',')
            array_float_format = []
            for no in range(0, len(array_string_format)):
                    array_float_format.append(float(array_string_format[no]))
            self.array_2d = numpy.append(self.array_2d, [array_float_format], axis=0)

    def standardise(self):
        no_of_rows = self.array_2d.shape[0]
        no_of_columns = self.array_2d.shape[1]

        # Get the mean value of each row and store in "mean_per_column_array" List
        mean_per_column_array = []
        for no in range(0, no_of_columns):
            mean_per_column_array.append(statistics.mean(self.array_2d[:, no]))

        # Get the max value from each column and store in "max_per_column_array" List
        max_per_column_array = []
        for no in range(0, no_of_columns):
            max_per_column_array.append(max(self.array_2d[:, no]))

        # Get the min value from each column and store in "min_per_column_array" List
        min_per_column_array = []
        for no in range(0, no_of_columns):
            min_per_column_array.append(min(self.array_2d[:, no]))
        
        # Algorithm to standardize array_2d
        counter = 0
        for row_no in range(0, no_of_rows):
            standardized_per_row_array = []
            for col_no in range(0, no_of_columns):
                numerator = self.array_2d[row_no, col_no] - mean_per_column_array[col_no]
                denominator = max_per_column_array[col_no] - min_per_column_array[col_no]
                standardized_form = numerator / denominator
                standardized_per_row_array.append(standardized_form)
            if counter == 0:
                standardized_array = numpy.array([standardized_per_row_array])
                counter +=1
                continue
            standardized_array = numpy.append(standardized_array, [standardized_per_row_array], axis=0)
        self.array_2d = standardized_array
    
    def get_distance(self, other_matrix, weights, beta):
        m = self.array_2d.shape[1]  # Set the no of column in the data_matrix (Standardized array_2d)
        n_row_om = other_matrix.shape[0]    # Set the no of rows in other_matrix (centroids)

        # Algorithm to calculate the Euclidean distance of a row in the data_matrix and all rows in the other_matrix(centroids)
        counter = 0
        for no_row in range(n_row_om):
            distance_total = 0
            for no_col in range(m):
                Euclidian_formular_result = (weights[no_col])**beta * (self.row_from_standardized_matrix[no_col] - other_matrix[no_row, no_col])**2
                distance_total += Euclidian_formular_result
            if counter == 0:
                distance_array = numpy.array([[distance_total]])
                counter += 1
                continue
            distance_array = numpy.append(distance_array, [[distance_total]], axis=0)

        return(distance_array)  # Return the distance array with K rows and 1 column
                


    def get_count_frequency(self):
        no_of_rows = self.array_2d.shape[0]
        no_of_columns = self.array_2d.shape[1]
        dictionary_mapping = {}

        if no_of_columns == 1:
            for row_no in range(0, no_of_rows):
                for col_no in range(0, no_of_columns):
                    if self.array_2d[row_no, col_no] in dictionary_mapping:
                        for value in dictionary_mapping:
                            if value == self.array_2d[row_no, col_no]:
                                dictionary_mapping[value] +=1
                    else:
                        dictionary_mapping[self.array_2d[row_no, col_no]] = 1
        return(dictionary_mapping)

def get_initial_weights(m):
    # Algorithm to derive a Randomly generated array with 1 row and m columns whose summation == 1

    # Get m randomly generated floating point numbers between 0 and 1 and store them in "random_values" List
    random_values = []
    for no in range(m):
        value = random.random()
        random_values.append(value)
    total = sum(random_values)

    # The summation of the divistion of a set of numbers by their total == 1. i.e total = no1 + no2; (no1/total) + (no2/total) == 1
    # Get the result of dividing each randomly generated number by their total and store each one in "final_random_values" List.
    final_random_values = []
    for no in range(m):
        final_random_values.append(random_values[no]/total)

    weights = numpy.array(final_random_values)  # Save the list in array called weigths
    return(weights) # Return the array


def get_centroids(data_matrix, S, K):
    no_rows_s = S.shape[0]
    counter = 0

    # Algorithm to get new values for centroids
    for no_rows in range(K):
        rows_equal_to_s = []
        for s_row in range(no_rows_s):
            if no_rows == S[s_row, 0]:
                rows_equal_to_s.append(s_row)
        counter1 = 0
        for no in rows_equal_to_s:
            if counter1 == 0:
                rows_from_data_matrix = numpy.array([data_matrix[no]])
                counter1 += 1
            else:
                rows_from_data_matrix = numpy.append(rows_from_data_matrix, [data_matrix[no]], axis=0)
        
        # Calculate mean of each column in array returned
        no_of_columns = rows_from_data_matrix.shape[1]
        mean_per_column_array = []
        for no in range(no_of_columns):
            mean_per_column_array.append(statistics.mean(rows_from_data_matrix[:, no]))
        
        # Save the mean values in each row i of the new centroids array
        if counter == 0:
            new_centroids = numpy.array([mean_per_column_array])
            counter += 1
            continue
        new_centroids = numpy.append(new_centroids, [mean_per_column_array], axis=0)
    
    # Return the new centroids array
    return(new_centroids)

        


def get_groups(m_matrix, K, beta):
    m_matrix.load_from_csv() # Standardize the csv file -- Data.csv
    
    global weightts
    global betaa
    betaa = beta  # Save beta in global betaa
    # print(betaa)

    m_matrix.standardise() # Standardize array_2d matrix
    data_matrix = m_matrix.array_2d  # Standardized form of original data in Data.csv

    # Get no of row (n) and no of columns (m)
    n = data_matrix.shape[0] # no of rows
    m = data_matrix.shape[1] # no of columns

    # Step 1: Positive value for K and Beta are set in the parameters.

    #Step 2 from clustering algorithm
    weights = get_initial_weights(m)
    weightts = weights  # Save weights in global weightts

    # step 3
    centroids = numpy.array([])

    # step 4
    S = numpy.array([[0]])
    for no in range(n-1):
        S = numpy.append(S, [[0]], axis=0)

    # step 5
    Last_row = n-1 
    counter = 0
    for no in range(K):
        if counter == 0:
            random_data_matrix = numpy.array([data_matrix[random.randint(0, Last_row)]])
            counter += 1
        else:
            random_data_matrix = numpy.append(random_data_matrix, [data_matrix[random.randint(0, Last_row)]], axis=0)
    
    # step 6
    centroids = random_data_matrix

    #Get my first set of values for weights
    weights_for_euclidean_formular = get_new_weights(data_matrix, centroids, S)

    # Step 7(a)
    for no_rows in range(n):
        m_matrix.row_from_standardized_matrix = data_matrix[no_rows]
        Euclidean_distance_array = m_matrix.get_distance(centroids, weights_for_euclidean_formular, beta)

        # step 7(b)
        normalize_to_one_row =  Euclidean_distance_array[:, 0]
        mininum_distance = min(normalize_to_one_row)
        no_of_columns = Euclidean_distance_array.shape[0]
        for col_no in range(no_of_columns):
            if normalize_to_one_row[col_no] == mininum_distance:
                index_no = col_no
                break
        
        # Get original value in S_i
        original_value = S[no_rows, 0]
        # Save index of the row with the closest distance to S array
        S[no_rows] = [index_no]

        # step 8
        if original_value == index_no:
            continue

        # Step 9
        centroids = get_centroids(data_matrix, S, K)

        # Step 10
        # Get new sets of values for weights
        weights_for_euclidean_formular = get_new_weights(data_matrix, centroids, S)

        # Step 11: End of loop, Goes back to step 7

    m_matrix.array_2d = S
    return(m_matrix)

def get_new_weights(data_matrix, centroids, S):
    m = data_matrix.shape[1]    # Get no of columns in the data_matrix
    K = centroids.shape[0]  # Get no of rows in centroids
    n = data_matrix.shape[0]    # Get the no of rows in the data_matrix
    counter = 0
    weights_for_euclidean_formular = numpy.array([])

    # Algorithm for calculating the weights that will be used in Euclidean formular
    for no_j in range(m):
        dispersion_j = 0
        for no_k in range(K):
            total = 0
            for no_i in range(n):
                standardized_row_col = data_matrix[no_i, no_j]   # Acronym = src
                centroids_value = centroids[no_k, no_j]          # Acronmy = cv
                subtraction_and_squared_of_src_cv = (standardized_row_col - centroids_value)**2
                if S[no_i, 0] == no_k:
                    u_i_k = 1
                else:
                    u_i_k = 0
                final_value = u_i_k * subtraction_and_squared_of_src_cv
                total += final_value
            
            dispersion_j += total
        if dispersion_j == 0:
            weight_j = 0
        else:
            total = 0
            for no_col in range(m):
                power_value = 1 / (betaa - 1)
                main_value = (dispersion_j / weightts[no_col])**power_value
                total += main_value
            weight_j = 1 / total
        
        if counter == 0:
            weights_for_euclidean_formular = numpy.array([[weight_j,]])
            counter += 1
            continue
        weights_for_euclidean_formular = numpy.append(weights_for_euclidean_formular, weight_j)

    return(weights_for_euclidean_formular)


        
def run_test():
    m = matrix('Data.csv')
    for k in range(2, 5):
        for beta in range(11, 25):
            S = get_groups(m, k, beta/10)
            print(str(k)+'-'+str(beta)+'='+str(S.get_count_frequency()))
    
run_test()