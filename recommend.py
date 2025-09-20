from random import sample

from abstractions import *
from data import ALL_RESTAURANTS, CATEGORIES, USER_FILES, load_user_file
from ucb import main
from utils import distance, mean, zip, enumerate
from visualize import draw_map

##################################
# Phase 2: Unsupervised Learning #
##################################


def find_closest(location, centroids):
    """Return the centroid in centroids that is closest to location.
    If multiple centroids are equally close, return the first one.

    >>> find_closest([3.0, 4.0], [[0.0, 0.0], [2.0, 3.0], [4.0, 3.0], [5.0, 5.0]])
    [2.0, 3.0]
    # in this case, location = [3.0, 4.0]
    # centroids = a list of centroid: [[0.0, 0.0], [2.0, 3.0], [4.0, 3.0], [5.0, 5.0]]
    """
    # BEGIN Question 3
    mn = distance(location, centroids[0])
    for centroid in centroids:
        tmp = distance(location, centroid)
        mn = min(mn, tmp)
    for centroid in centroids:
        if distance(location, centroid)==mn:
            return centroid
    # END Question 3


def group_by_key(pairs):
    """Given a list of lists, where each inner list is a [key, value] pair,
    return a new list that groups values by their key.

    Arguments:
    pairs -- a sequence of [key, value] pairs

    >>> example = [ [1, 2], [3, 2], [2, 4], [1, 3], [3, 1], [1, 2] ]
    >>> group_by_key(example)
    [[2, 3, 2], [2, 1], [4]]
    """
    keys = []
    for key, _ in pairs:
        if key not in keys:
            keys.append(key)
    # keys = a list of key in *pairs*
    return [[v for k, v in pairs if k == key] for key in keys]
    # group_by_key returns a list of lists, each small list contains all the values
    # that have the same key


def group_by_centroid(restaurants, centroids):
    """Return a list of clusters, where each cluster contains all restaurants
    nearest to a corresponding centroid in centroids. Each item in
    restaurants should appear once in the result, along with the other
    restaurants closest to the same centroid.
    >>> r1 = make_restaurant('X', [4, 3], [], 3, [
    ...         make_review('X', 4.5),
    ...      ]) # r1's location is [4,3]
    >>> r2 = make_restaurant('Y', [-2, -4], [], 4, [
    ...         make_review('Y', 3),
    ...         make_review('Y', 5),
    ...      ]) # r2's location is [-2, -4]
    >>> r3 = make_restaurant('Z', [-1, 2], [], 2, [
    ...         make_review('Z', 4)
    ...      ]) # r3's location is [-1, 2]
    >>> c1 = [4, 5]
    >>> c2 = [0, 0]
    >>> groups = group_by_centroid([r1, r2, r3], [c1, c2])
    >>> [[restaurant_name(r) for r in g] for g in groups]
    [['X'], ['Y', 'Z']] # r1 is closest to c1, r2 and r3 are closer to c2
    """
    # returns a list of clusters, each cluster is a group of restaurants that are closer 
    # specific centroid in centroids

    # BEGIN Question 4
    group = []
    for restaurant in restaurants:
        tmp = restaurant_location(restaurant) # tmp = [x, y]
        entry = [find_closest(tmp, centroids), restaurant]
        group. append(entry)
    return group_by_key(group)
    # END Question 4


def find_centroid(cluster):
    """Return the centroid of the locations of the restaurants in cluster.
    >>> r1 = make_restaurant('X', [4, 3], [], 3, [
    ...         make_review('X', 4.5),
    ...      ]) # r1's location is [4,3]
    >>> r2 = make_restaurant('Y', [-3, 1], [], 4, [
    ...         make_review('Y', 3),
    ...         make_review('Y', 5),
    ...      ]) # r2's location is [-3, 1]
    >>> r3 = make_restaurant('Z', [-1, 2], [], 2, [
    ...         make_review('Z', 4)
    ...      ]) # r3's location is [-1, 2]
    >>> cluster = [r1, r2, r3]
    >>> find_centroid(cluster)
    [0.0, 2.0]
    """
    # It takes in a list of restaurants (restaurants) 
    # BEGIN Question 5
    latitude = []
    longitude = []
    for restaurant in cluster:
        a = restaurant_location(restaurant)
        latitude. append(a[0])
        longitude. append(a[1])
    return [mean(latitude), mean(longitude)]
    # END Question 5


def k_means(restaurants, k, max_updates=100):
    """Use k-means to group restaurants by location into k clusters."""
    assert len(restaurants) >= k, 'Not enough restaurants to cluster'
    previous_centroids = []
    n = 0
    # Select initial centroids randomly by choosing k different restaurants
    centroids = [restaurant_location(r) for r in sample(restaurants, k)]

    while previous_centroids != centroids and n < max_updates:
        previous_centroids = centroids
        # BEGIN Question 6
        # Step 1:   
        clusters = group_by_centroid(restaurants, centroids)
        # Step 2
        centroids = [find_centroid(cluster) for cluster in clusters]
        # END Question 6
        n += 1
    return centroids


def find_predictor(user, restaurants, feature_fn):
    """Return a score predictor (a function that takes in a restaurant
    and returns a predicted score) for a user by performing least-squares
    linear regression using feature_fn on the items in restaurants.
    Also, return the R^2 value of this model.

    Arguments:
    user -- A user # data abstraction
    restaurants -- A sequence of restaurants
    feature_fn -- A function that takes a restaurant and returns a number
    """
    # feature_fn: 
    # A single-argument function that takes a restaurant
    # and returns a number representing a feature of the restaurant, such as its mean score or price


    # Dictionary comprehension (very similar to list comprehension)
    # that creates a dictionary, reviews_by_user, where the key
    # is the name of the restaurant the user reviewed, and the value
    # is the review score for that restaurant.
    reviews_by_user = {review_restaurant_name(review): review_score(review)
                       for review in user_reviews(user).values()}

    xs = [feature_fn(r) for r in restaurants]
    ys = [reviews_by_user[restaurant_name(r)] for r in restaurants]
    #ys is a list of review score for each restaurant in *restaurants*


    # BEGIN Question 7
    # b, a, r_squared = 0, 0, 0 
    S_xx, S_yy, S_xy = 0, 0, 0
    for i in range(len(xs)):
        S_xx += ((xs[i] - mean(xs))**2)
        S_yy += ((ys[i] - mean(ys))**2)
        S_xy += (xs[i] - mean(xs))* (ys[i] - mean(ys))
    b = S_xy / S_xx
    a = mean(ys) - b* mean(xs)
    r_squared = (S_xy**2)/(S_xx * S_yy)
    # END Question 7
    def predictor(restaurant):
        return b * feature_fn(restaurant) + a

    return predictor, r_squared


def best_predictor(user, restaurants, feature_fns):
    """Find the feature within feature_fns that gives the highest R^2 value
    for predicting scores by the user; return a predictor using that feature.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of functions that each takes a restaurant
    """
    reviewed = user_reviewed_restaurants(user, restaurants)
    # BEGIN Question 8
    dict = {}
    for func in feature_fns:
        predictor, r = find_predictor(user, reviewed, func)
        dict[predictor] = r
    return max(dict, key = lambda x: dict[x])
    # END Question 8


def rate_all(user, restaurants, feature_fns):
    """Return the predicted scores of restaurants by user using the best
    predictor based on a function from feature_fns.

    Arguments:
    user -- A user
    restaurants -- A list of restaurants
    feature_fns -- A sequence of feature functions
    """
    predictor = best_predictor(user, ALL_RESTAURANTS, feature_fns)
    reviewed = user_reviewed_restaurants(user, restaurants)
    # BEGIN Question 9
    dict = {}
    for restaurant in restaurants:
        if restaurant in reviewed:
            dict[restaurant_name(restaurant)] = user_score(user, restaurant_name(restaurant))
        else:
            dict[restaurant_name(restaurant)] = predictor(restaurant)
    return dict
    # END Question 9


def search(query, restaurants):
    """Return each restaurant in restaurants that has query as a category.

    Arguments:
    query -- A string
    restaurants -- A sequence of restaurants
    """
    # BEGIN Question 10
    lst = [restaurant for restaurant in restaurants if query in restaurant_categories(restaurant)]
    return lst
    # END Question 10


def feature_set():
    """Return a sequence of feature functions."""
    return [restaurant_mean_score,
            restaurant_price,
            restaurant_num_scores,
            lambda r: restaurant_location(r)[0],
            lambda r: restaurant_location(r)[1]]


@main
def main(*args):
    import argparse
    parser = argparse.ArgumentParser(
        description='Run Recommendations',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--user', type=str, choices=USER_FILES,
                        default='test_user',
                        metavar='USER',
                        help='user file, e.g.\n' +
                        '{{{}}}'.format(','.join(sample(USER_FILES, 3))))
    parser.add_argument('-k', '--k', type=int, help='for k-means')
    parser.add_argument('-q', '--query', choices=CATEGORIES,
                        metavar='QUERY',
                        help='search for restaurants by category e.g.\n'
                        '{{{}}}'.format(','.join(sample(list(CATEGORIES), 3))))
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict scores for all restaurants')
    parser.add_argument('-r', '--restaurants', action='store_true',
                        help='outputs a list of restaurant names')
    args = parser.parse_args()

    # Output a list of restaurant names
    if args.restaurants:
        print('Restaurant names:')
        for restaurant in sorted(ALL_RESTAURANTS, key=restaurant_name):
            print(repr(restaurant_name(restaurant)))
        exit(0)

    # Select restaurants using a category query
    if args.query:
        restaurants = search(args.query, ALL_RESTAURANTS)
    else:
        restaurants = ALL_RESTAURANTS

    # Load a user
    assert args.user, 'A --user is required to draw a map'
    user = load_user_file('{}.dat'.format(args.user))

    # Collect ratings
    if args.predict:
        ratings = rate_all(user, restaurants, feature_set())
    else:
        restaurants = user_reviewed_restaurants(user, restaurants)
        names = [restaurant_name(r) for r in restaurants]
        ratings = {name: user_score(user, name) for name in names}

    # Draw the visualization
    if args.k:
        centroids = k_means(restaurants, min(args.k, len(restaurants)))
    else:
        centroids = [restaurant_location(r) for r in restaurants]
    draw_map(centroids, restaurants, ratings)
