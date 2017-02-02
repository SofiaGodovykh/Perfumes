import json
import cPickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import datetime
from pymongo import MongoClient
from sklearn.linear_model import SGDClassifier


def get_all_perfumes_from_db():
    """
    :return: MongoDB cursor
    """
    client = MongoClient('localhost', 27017)
    db = client['fragrantica']
    collection = db['perfumes']
    all_perfumes = collection.find()
    return all_perfumes


def ids_notes(perfumes):
    """
    :param perfumes: list of perfumes
    :return: dictionary where id is a key and note is a value
    """
    ids_notes = dict()
    for perfume in perfumes:
        for pos in ['voted', 'top', 'middle', 'base']:
            if len(perfume['notes'][pos]) != 0:
                for note in perfume['notes'][pos]:
                    name = note['name']
                    id = int(note['id'])
                    ids_notes.update({id: name})

    with open('ids_notes', 'wb+') as outfile:
        json.dump(ids_notes, outfile)

    return ids_notes


def positions_names(ids_notes):
    """
    :param ids_notes: dictionary where id is a key and note is a value
    :return: array of features labels
    """
    labels = list()
    ids = ids_notes.keys()
    ids.sort()
    for i in ids:
        labels.append(ids_notes[i])

    return labels


def transform_ids_to_indexes(ids_notes):
    """
    makes an dictionary {id : position in an features array}
    :param ids_notes:
    :return:
    """
    ids = ids_notes.keys()
    ids.sort()
    data = dict()
    for i in range(len(ids)):
        data.update({ids[i]: i})
    return data


def make_features_array(perfume, ids_to_indexes, class_dictionary):
    """
    forms an array of binary features
    :param class_dictionary: dictionary where perfume name is a key and perfume group id is a value,
    result of perfume_group_target() or perfume_group_target_reduced()
    :param perfume: perfume from db
    :param ids_to_indexes: dict with ids and corresponding positions in a features vector
    :return: binary features array
    """
    data = [0] * len(ids_to_indexes)
    for pos in ['voted', 'top', 'middle', 'base']:
        if len(perfume['notes'][pos]) != 0:
            for note in perfume['notes'][pos]:
                id = int(note['id'])
                try:
                    data[ids_to_indexes[id]] = 1
                except:
                    continue

    data = (data, class_dictionary[perfume['group']['name']])
    return data


def fit_classifier(data, save, target):
    """
    :param data: features array
    :param save: True - save classifier
    :param target:
    :return: fitter classifier
    """
    classifier = SGDClassifier(penalty='l2', alpha=0.001, loss='log', fit_intercept=True)
    scores = cross_val_score(classifier, data, target, cv=5)
    print sum(scores) / len(scores)
    classifier.fit(data, target)
    if save:
        with open(str(datetime.datetime.now()), 'wb') as f:
            cPickle.dump(classifier, f)

    return classifier


def find_best_parameters(classifier, data, target):
    """
    prints GridSearchCV results
    """
    params = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber',
                       'epsilon_insensitive', 'squared_epsilon_insensitive'],
              'penalty': ['none', 'l2', 'l1', 'elasticnet'],
              'alpha': (1e-2, 1e-3), 'fit_intercept': [True, False]}
    clf = GridSearchCV(classifier, params, cv=5, verbose=10)

    clf.fit(data, target)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))


def perfume_group_target():
    """
    :return: dictionary where perfume name is a key and perfume group (citrus, floral, etc.) integer id is a value
    """
    perfumes = get_all_perfumes_from_db()
    target_labels = set()
    for p in perfumes:
        target_labels.add(p['group']['name'])

    target_labels = list(target_labels)
    classes = dict()
    for i in range(len(target_labels)):
        classes.update({target_labels[i]: i})

    classes.update({'': -1})
    return classes


def perfume_group_target_reduced():
    """
    :return: dictionary where perfume name is a key and perfume group (citrus, floral, etc.) integer id is a value,
    where number of groupes is limited by 7
    """
    classes = dict()
    classes.update(
        {'': -1, 'Aromatic': 0, 'Aromatic Aquatic': 0, 'Aromatic Fougere': 0, 'Aromatic Fruity': 0, 'Aromatic Green': 0,
         'Aromatic Spicy': 0})
    classes.update({'Chypre': 1, 'Chypre Floral': 1, 'Chypre Fruity': 1})
    classes.update({'Citrus': 2, 'Citrus Aromatic': 2, 'Citrus Gourmand': 2})
    classes.update(
        {'Floral Aldehyde': 3, 'Floral Aquatic': 3, 'Floral Fruity': 3, 'Floral Fruity Gourmand': 3, 'Floral Green': 3,
         'Floral Woody Musk': 3, 'Floral': 3})
    classes.update({'Leather': 4})
    classes.update(
        {'Oriental Floral': 5, 'Oriental Fougere': 5, 'Oriental Spicy': 5, 'Oriental Vanilla': 5, 'Oriental Woody': 5,
         'Oriental': 5})
    classes.update(
        {'Woody Aquatic': 6, 'Woody Aromatic': 6, 'Woody Chypre': 6, 'Woody Floral Musk': 6, 'Woody Spicy': 6,
         'Woody': 6})
    return classes


def get_gender(perfume):
    """
    :param perfume:
    :return: integer target class, where 0 is women, 1 is men, 2 is unisex, the rest is -1
    """
    name = perfume['name'].lower()
    if 'and' in name or 'unisex' in name:
        return 2
    if 'woman' in name or 'women' in name or 'femme' in name:
        return 0
    if 'man' in name or 'men' in name or 'homme' in name:
        return 1
    return -1


def print_top(clf, labels, n):
    """
    :param labels:
    :param clf: fitted classifier
    :param n: number of printed features, int
    :return: prints the most important features with weights
    """
    coefs = sorted(zip(clf.coef_[0], labels))
    top = zip(coefs[:n], coefs[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

def get_time_to_wear(perfume, ratio = 0.5):
    """
    :param perfume: mongodb perfume
    :param ratio: the minimum value of min(day/night, night/day)/max(day/night, night/day) to measure the difference
    :return: 0 night, 1 day, -1 both or unknown
    """
    day = perfume['day']
    night = perfume['night']
    if day * night != 0 and (1.0 * day / night <= ratio or 1.0 * night / day <= ratio):
        if day > night:
            return 1
        else:
            return 0
    if day * night == 0:
        if day > night:
            return 1
        else:
            return 0
    return -1

def get_temperature(perfume, ratio = 0.5):
    """
    :param perfume: mongodb perfume
    :param ratio: the minimum value of min(day/night, night/day)/max(day/night, night/day) to measure the difference
    :return: 0 night, 1 day, -1 both or unknown
    """
    hot = perfume['hot']
    cold = perfume['cold']
    if hot * cold != 0 and (1.0 * hot / cold <= ratio or 1.0 * cold / hot <= ratio):
        if hot > cold:
            return 1
        else:
            return 0
    if hot * cold == 0:
        if hot > cold:
            return 1
        else:
            return 0
    return -1