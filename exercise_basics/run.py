import numpy as np


# numeric
def add(a: float, b: float):
    """Add a to b and return result."""
    # TODO: Implement this method
    return a + b
    raise NotImplementedError()


def power(a: float, e: float):
    """Returns a to the power of e."""
    # TODO: Implement this method
    return a ** e 
    raise NotImplementedError()


def modulo(a: int, b: int):
    """Returns a modulo b."""
    # TODO: Implement this method
    return a % b
    raise NotImplementedError()


# strings
def string_len(string: str):
    """Returns length of string."""
    # TODO: Implement this method
    return len(string)
    raise NotImplementedError()


def char_at(string: str, i: int):
    """Returns character at position i of string."""
    # TODO: Implement this method
    return string[i]
    raise NotImplementedError()


def get_sub_string(string: str, start: int, end: int):
    """Returns substring from index start to end of string."""
    # TODO: Implement this method
    return string[start:end]
    raise NotImplementedError()


def contains_sub_string(string: str, sub_string: str):
    """Returns true if sub_string is part of string."""
    # TODO: Implement this method 
    if sub_string in string:
        return True
    else:
        return False
    raise NotImplementedError()


def get_string_date(day: int, month: int, year: int):
    """Returns the date as a string in the format DD-MM-YYYY.
    Example: 13-01-2020"""
    # TODO: Implement this method
    return f"{day:02d}-{month:02d}-{year}"
    raise NotImplementedError()


# list
def list_len(a: list):
    """Returns the length of a list."""
    # TODO: Implement this method
    return len(a)
    raise NotImplementedError()


def last_element(a: list):
    """Returns the last element of a list.
    The list won't be empty."""
    # TODO: Implement this method
    return a[-1]
    raise NotImplementedError()


def contains_element(a: list, e):
    """Returns True if e is an element of list a."""
    # TODO: Implement this method
    if e in a:
        return True
    else: 
        return False
    raise NotImplementedError()


def sum_(a: list):
    """Returns the sum of all elements of list a.
    List a only contains integers and floats."""
    # TODO: Implement this method
    return sum(a)
    raise NotImplementedError()


def mean(a: list):
    """Returns the mean over all elements of list a.
    List a only contains integers and floats."""
    # TODO: Implement this method
    return sum(a) / len(a)
    raise NotImplementedError()


def pairwise_add(a: list, b: list):
    """Returns a list where the i-th element is the sum of the i-th element of a and b.
    List a and b only contain integers and floats.
    List a and b have the same length.
    Hint: take a look at the in-built python function zip()"""
    # TODO: Implement this method
    ab = []
    for a, b in zip(a, b):
        ab.append(a + b)
    return ab    
    raise NotImplementedError()

def third_reverse(a: list):
    """ Returns a list containing every third element but in inverted order.
    """
    # TODO: Implement this method
    third_list = [] * len(a % 3)
    for i, element in enumerate(a):
        if i % 3 == 2:
            third_list.append(a[i])
    return third_list[::-1] # kehrt die liste um
    # noch eine Möglichkeit: return reversed(third_list)
    raise NotImplementedError()


# numpy
def np_pairwise_add(a: np.ndarray, b: np.ndarray):
    """Returns a numpy array where the i-th element is the sum of the i-th element of a and b.
    Numpy array a and b only contain integers and floats.
    Numpy array a and b have the same shape."""
    # TODO: Implement this method
    usefullArray = np.zeros(len(a))
    for i, (a_elem, b_elem) in zip(a, b): 
        list.append(a_elem + b_elem)
    return usefullArray    
    raise NotImplementedError()


def np_get_column_at(a: np.ndarray, i: int):
    """Returns the i-th column of the matrix a.
    a has at least 2 dimensions; shape (M, N, ...)"""
    # TODO: Implement this method
    return a[:, i]
    raise NotImplementedError()


def np_index_reverse(a: np.ndarray, i: int):
    """ Returns a numpy array containing every i-th element but in inverted order.
    """
    # TODO: Implement this method
    usefullArray_length = len(a) // i # // ist für eine ganzzahl rückgabe
    usefullArray = np.zeros(usefullArray_length)
    k = 0
    for ii, element in enumerate(a):
        if ii % i == i - 1:
            usefullArray[k] = element
            k += 1
    return usefullArray[::-1]       
    raise NotImplementedError()


def np_mean(a: np.ndarray):
    """Returns the mean over all elements of numpy array a regardless of shape."""
    # TODO: Implement this method
    mean_digit = 0
    elements_in_array = 0
    for element in a:
        mean_digit += element
        elements_in_array += 1
    return mean_digit / elements_in_array
    raise NotImplementedError()


def np_mean_per_row(a: np.ndarray): # https://stackoverflow.com/questions/18688948/numpy-how-do-i-find-total-rows-in-a-2d-array-and-total-column-in-a-1d-array
    """Returns the mean for each row of matrix a.
    a has at least 2 dimensions; shape (M, N, ...)."""
    # TODO: Implement this method
    return np.mean(a, axis=1)
    raise NotImplementedError()


def np_mean_per_column(a: np.ndarray):
    """Returns the mean for each column of matrix a.
    a has at least 2 dimensions; shape (M, N, ...)."""
    # TODO: Implement this method
    return np.mean(a, axis=0)
    raise NotImplementedError()


def np_to_row_vector(a: np.ndarray):
    """Returns a numpy array of shape (1, N) where N is the number of elements in a.
    a will always be of shape (N)."""
    # TODO: Implement this method
    return a.reshape(1, -1)
    raise NotImplementedError()


def np_row_vector_to_column_vector(a: np.ndarray):
    """Returns a numpy array of shape (N, 1).
     a will always be a row vector with shape (1, N)."""
    # TODO: Implement this method
    return a.reshape(-1, 1)
    raise NotImplementedError()


def np_column_vector_to_row_vector(a: np.ndarray):
    """Returns a numpy array of shape (1, N).
    a will always be a row vector with shape (N, 1)."""
    # TODO: Implement this method
    return a.reshape(1, -1)
    raise NotImplementedError()


def np_auto_column_row_vector_conversion(a: np.ndarray):
    """Returns a numpy array of shape (1, N) if a is of shape (N, 1)
    or a numpy array of shape (N, 1) if a is of shape (1, N).
    a will always be of shape (N, 1) or (1, N).
    Hint: take a look at matrix operation transpose https://en.wikipedia.org/wiki/Transpose"""
    # TODO: Implement this method
    if a.shape[0] == 1:  # Zeilenvektor zu Spaltenvektor
        return a.reshape(-1, 1)
    else:  # Spaltenvektor zu Zeilenvektor
        return a.reshape(1, -1)
    raise NotImplementedError()


def np_dot_product(a: np.ndarray, b: np.ndarray):
    """Returns the scalar product (also called: dot product) of a and b.
    a and b are numpy arrays of shape (N)."""
    # TODO: Implement this method
    return np.dot(a, b)
    raise NotImplementedError()


def np_matrix_product(a: np.ndarray, b: np.ndarray):
    """Returns the matrix product of matrices a and b.
    a is a numpy array of shape (N, M).
    b is a numpy array of shape (M, N)."""
    # TODO: Implement this method
    return np.matmul(a, b)
    raise NotImplementedError()


if __name__ == "__main__":
    a = np.arange(1, 10)
    b = np.arange(6, 15)
    c = np.arange(1, 10).reshape(3, 3)

    print("--- numeric ---\n")
    print("add: ", add(5.7, 1.3))
    print("power: ", power(2, 3))
    print("modulo: ", modulo(18, 4))
    print("\n")

    print("--- strings ---\n")
    print("string_len: ", string_len("hello"))
    print("char_at: ", char_at("hello", 1))
    print("get_sub_string: ", get_sub_string("hello", 2, 4))
    print("contains_sub_string: ", contains_sub_string("hello", "ll"))
    print("get_string_date: ", get_string_date(28, 4, 2021))
    print("\n")

    print("--- list ---\n")
    print("list_len: ", list_len([1, 2, 3]))
    print("last_element: ", last_element([1, 2, 1]))
    print("contains_element: ", contains_element([1, 2, 7], 7))
    print("sum_: ", sum_([1, 2, 3]))
    print("mean: ", mean([1, 2, 3]))
    print("pairwise_add: ", pairwise_add([1, 2, 3], [4, 5, 6]))
    print("third_reverse: ", third_reverse([1, 2, 3, 4, 5, 6]))
    print("\n")

    print("--- numpy ---\n")
    print("pairwise_add: ", np_pairwise_add(a, b))
    print("np_get_column_at: ", np_get_column_at(c, 1))
    print("np_index_reverse: ", np_index_reverse(a, 3))
    print("np_mean: ", np_mean(c))
    print("np_mean_per_row: ", np_mean_per_row(c))
    print("np_mean_per_column: ", np_mean_per_column(c))
    aRV = np_to_row_vector(a)
    print("np_to_row_vector: ", aRV)
    aCV = np_row_vector_to_column_vector(aRV)
    print("np_row_vector_to_column_vector: ", aCV)
    print("np_column_vector_to_row_vector: ", np_column_vector_to_row_vector(aCV))
    print("np_auto_column_row_vector_conversion: ", np_auto_column_row_vector_conversion(aRV))
    print("np_auto_column_row_vector_conversion: ", np_auto_column_row_vector_conversion(aCV))
    print("np_dot_product: ", np_dot_product(a, b))
    print("np_matrix_product: ", np_matrix_product(aRV, aCV))
