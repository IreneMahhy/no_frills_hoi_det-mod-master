import numpy as np
import matplotlib.pyplot as plt


def create_confusion_matrix(num_categories, predicted_categories, categories, test_labels, categories_name, n_pos, norm=False):
    """

    :param num_categories: The number of categories, in our case, it should be 29.
    :param predicted_categories: The predicted result, which denotes every action category predicted from HO pair. [n,]
    :param categories: The categories ID. [29,]
    :param test_labels: [n,]
    :param categories_name: []
    :param n_pos: The ground truth num of each category. [29,]
    :return:
    """
    # Initialize the matrix
    confusion_matrix = np.zeros((num_categories, num_categories))

    # Iterate over predicted results (this is like, several hundred items long)
    for i, cat in enumerate(predicted_categories):
        # Find the row and column corresponding to the label of this entry
        # The row is the ground truth label and the column is the found label
        row = np.argwhere(categories == test_labels[i])[0][0]
        column = np.argwhere(categories == predicted_categories[i])[0][0]

        # This way we build up a histogram from our labeled data
        confusion_matrix[row][column] += 1

    if norm:
        n_pos = np.array(n_pos)
        n_pos_expand = np.expand_dims(n_pos, 1)
        confusion_matrix = confusion_matrix / n_pos_expand
    num_test_per_cat = len(test_labels) / num_categories
    confusion_matrix = confusion_matrix / float(num_test_per_cat)
    accuracy = np.mean(np.diag(confusion_matrix))

    plt.imshow(confusion_matrix, cmap='plasma', interpolation='nearest')
    # We put the shortened labels (e.g. "sub" for "suburb") on the x axis
    locs, labels = plt.xticks()
    plt.xticks(np.arange(num_categories), categories_name, rotation=-90)
    plt.xlabel('predicted results')
    # Full labels go on y
    locs, labels = plt.yticks()
    plt.yticks(np.arange(num_categories), categories_name)
    plt.ylabel('ground truth')
    # Save the result
    plt.savefig('confusion_matrix.png', bbox_inches='tight')


if __name__ == '__main__':
    num_categories = 5
    test_labels = np.random.permutation(5)
    random_permutation = np.random.permutation(len(test_labels))
    predicted_categories = [test_labels[i] for i in random_permutation]
    categories = [0, 1, 2, 3, 4]
    categories_name = ['hold', 'cut', 'shake', 'pick', 'hit']
    n_pos = [2, 3, 3, 2, 3]
    create_confusion_matrix(num_categories, predicted_categories, categories, test_labels, categories_name, n_pos)
