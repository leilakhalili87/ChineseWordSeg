import numpy as np
from scipy.sparse import csr_matrix

n_gram = 4
vec_size = 5


def read_file(file_name, encoding='big5hkscs'):
    """
    This function reads the input file.

    Parameter
    ---------
    file_name: str
        The name of the input file
    encoding: str
        The encoding method

    Return
    ------
    parsed: list
        The parsed input file

    """
    parsed = []
    with open(file_name, encoding=encoding, errors='ignore') as f:
        for line in f:
            parsed.append('\t' + line)
    return parsed


def remove_space(string):
    """
    This function removes the spaces in each line.

    Parameters
    ----------
    string: str
        The input line

    Return
    -------
    string: str
        The string with no space.
    """
    return string.replace(" ", "")


def create_corpus(parsed):
    """
    This function creates the corpus.

    Parameters
    ----------
    parsed:
        The parsed line of the input file

    Return
    -------
    corpus: dict
        The corpus dictionary
    """
    counter = 0
    corpus = {}
    for lines in parsed:
        new_str = remove_space(lines)
        for char in range(len(new_str)):
            corpus[new_str[char]] = 0
            if new_str[char] != '\n':
                corpus[new_str[char] + new_str[char+1]] = 0
    for key in corpus:
        corpus[key] = counter
        counter += 1
    # adding out of vocabulary to catch the new characters in test dataset
    corpus['oov'] = counter
    return corpus


def find_4_char(sentence, index, corpus):
    """
    This function creates one 4-grams and the label corresponding
    to the 4-gram.

    Parameter
    ---------
    sentence: str
        A line in the input file
    index: int
        The index of the 1st character of the 4-gram
    corpus: dict
        A dictionary of all the words in the corpus

    Return
    ------
    fv: list
        feature vector of a single 4-gram.
        This list has 5 elements which are the indecis of the characters
        in the corpus.
        eg: 4_gram = ABCD ==> feature vector=(AB, B, BC, C, CD)
    label: int
        The label is either 0 or 1.
    """
    num_char = 0
    output = ""
    for num in sentence[index:]:
        if num != " ":  # num_char increases for non-space charecters
            num_char += 1
        output += num
        if num_char == 4:  # just consider 4 characters at each iteration
            # There is/isn't a space at the middle of extracted characters
            if output[2] == " ":
                label = 1  
            else:
                label = 0
              
            no_space = remove_space(output)
            #  feature vector
            UniBiGram = [no_space[0:2], no_space[1], no_space[1:3], no_space[2], no_space[2:4]]
            # for items not existing in the corpus return the index of 'oov'
            fv = [corpus.get(key, corpus['oov']) for key in UniBiGram]

            return fv, label
    return [], []


def fv_sen(parsed, corpus):
    """
    This function finds the indecis of rows and columns of non-zero values
    in the featur vector.

    Parameter
    ---------
    parsed: list
        The parsed input file

    corpus: dict
        The corpus dictionary

    Return
    ------
    row: list
        A 1*5 list containing the line number in the parsed text.
    column: list
        A 1*5 list containing the values of the elements in the
        feature vector in the corpus dictionary.
    Y_train: list
        The label of the all the train/test dataset.

    """
    row = []
    Y_train = []
    column = []
    for numer in range(len(parsed)):
        sentence = parsed[numer]
        line = remove_space(sentence)
        if len(line) > n_gram:
            # the number of possible n_grams is len(sentence) - (n_gram-1)
            for index in range(len(sentence) - (n_gram-1)):
                if sentence[index] != " ":  #  ignore the spaces
                    fv, label = find_4_char(sentence, index, corpus)
                    if len(fv) != 0:  #  deal with None values
                        column.append(fv)
                        row.append([numer]*vec_size)
                        Y_train.append(label)
    return row, column, Y_train


def sparse_generator(row, column, n_corpus):
    """
    The function generates sparse matric for feature vector.
    
    Parameter
    ---------
    row: list
        A 1*5 list containing the line number in the parsed text.
    column: list
        A 1*5 list containing the values of the elements in the
        feature vector in the corpus dictionary.
    n_corpus: int
        number of the items in the corpus

    Return
    ------
    X: scipy csr matrix
        feature vector
    """
    num_4gram, n_featur = np.shape(row)
    # reshape row and column to size (num_4gram, n_featur)
    ROW = np.reshape((row), -1)
    COLUMN = np.reshape((column), -1)
    data = np.ones(n_featur * num_4gram)
    X = csr_matrix((data, (ROW, COLUMN)), shape=(num_4gram, n_corpus))
    return X


def Train_XY(Path2Data, FileTrain):
    """
    The function generates corpus, feature vector and labels for
    training dataset.
    
    Parameter
    ---------
    Path2Data: str
        Path to the input folder
    FileTrain: str
        name of the training dataset file

    Return
    ------
    corpus: dict
        The corpus dictionary
    X_train: scipy csr matrix
        feature vector for training dataset
    Y_train: list
        labels for the training dataset
    """
    parsed_train = read_file(Path2Data + FileTrain)
    corpus = create_corpus(parsed_train)
    n_corpus = len(corpus)
    row, column, Y_train = fv_sen(parsed_train, corpus)
    # generates a sparse matrix for feature vector of training dataset
    X_train = sparse_generator(row, column, n_corpus)
    return corpus, X_train, Y_train

def Test_XY(Path2Data, FileTest, corpus):
    """
    The function generates feature vector and labels for test dataset.
    
    Parameter
    ---------
    Path2Data: str
        Path to the input folder
    FileTrain: str
        name of the test dataset file

    Return
    ------
    corpus: dict
        The corpus dictionary
    X_test: scipy csr matrix
        feature vector for test dataset
    Y_test: list
        labels for the test dataset
    """
    parsed_test = read_file(Path2Data + FileTest)
    row_test, column_test, Y_test = fv_sen(parsed_test, corpus)
    n_corpus = len(corpus)
    X_test = sparse_generator(row_test, column_test, n_corpus)
    return X_test, Y_test