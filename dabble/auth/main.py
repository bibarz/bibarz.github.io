import csv
import numpy
import auth

'''
User authentication from keypad patterns. Data consists of:
    - User Id
    - Device screen height
    - Device screen width
    - Unix timestamp
    - 7 instances (7 keystrokes) consisting each of 10 features:
        - touch down time
        - touch down x coordinate
        - touch down y coordinate
        - touch down size
        - touch down pressure
        - touch up time
        - touch up x coordinate
        - touch up y coordinate
        - touch up size
        - touch up pressure
'''

def main(data_training_file_name, data_testing_file_name):
    data_training_file = open(data_training_file_name, 'rb')
    csv_training_reader = csv.reader(data_training_file, delimiter=',', quotechar='"')
    csv_training_reader.next()

    training_dataset = dict()

    for row in csv_training_reader:
        if row[0] not in training_dataset:
            training_dataset[row[0]] = numpy.array([]).reshape((0, len(row[1:])))
        training_dataset[row[0]] = numpy.vstack([training_dataset[row[0]], numpy.array(row[1:]).astype(float)])

    data_testing_file = open(data_testing_file_name, 'rb')
    csv_testing_reader = csv.reader(data_testing_file, delimiter=',', quotechar='"')
    csv_testing_reader.next()

    testing_dataset = dict()

    for row in csv_testing_reader:
        if row[0] not in testing_dataset:
            testing_dataset[row[0]] = numpy.array([]).reshape((0, len(row[1:])))
        testing_dataset[row[0]] = numpy.vstack([testing_dataset[row[0]], numpy.array(row[1:]).astype(float)])

    templates = dict()

    # For each user build a template
    for u in training_dataset:
        templates[u] = auth.build_template(u, training_dataset)

    # For each user test authentication
    all_false_accept = 0
    all_false_reject = 0
    all_true_accept = 0
    all_true_reject = 0
    for u in training_dataset:

        # Test false rejections
        true_accept = 0
        false_reject = 0
        for instance in testing_dataset[u]:
            (score, threshold) = auth.authenticate(instance, u, templates)
            # If score higher than the threshold, we accept the user as authentic
            if score > threshold:
                true_accept += 1
            else:
                false_reject += 1

        # Test false acceptance
        true_reject = 0
        false_accept = 0
        for u_attacker in testing_dataset:
            if u == u_attacker:
                continue
            for instance in testing_dataset[u_attacker]:
                (score, threshold) = auth.authenticate(instance, u, templates)
                # If score lower of equal to the threshold, we reject the user as an attacker
                if score <= threshold:
                    true_reject += 1
                else:
                    false_accept += 1
                    print('%s: bad attack from %s' % (u, u_attacker))
        print "--------------------------------------------------------------------"
        print "For user " + u + ":"
        print "  True Accepts: " + str(true_accept)
        print "  False Rejects: " + str(false_reject)
        print "  True Rejects: " + str(true_reject)
        print "  False Accepts: " + str(false_accept)
        all_false_accept += false_accept
        all_false_reject += false_reject
        all_true_accept += true_accept
        all_true_reject += true_reject

    print "bad reject rate: %.0f, bad accept rate: %.0f" % (100. * float(all_false_reject) / (all_false_reject + all_true_accept),
                                                            100. * float(all_false_accept) / (all_false_accept + all_true_reject))


if __name__ == "__main__":
    main('dataset_training.csv', 'dataset_testing.csv')
