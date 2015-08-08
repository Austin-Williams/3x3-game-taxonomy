#

import itertools as it
import numpy as np
import pickle
import os.path

def generateBucket(b):
    global player_switch_templates
    global templates
    #This will output the bth bucket of bimatrices.
    #Should be the case that 0 <= b < 50,808,240
    player_one_template = np.array(templates[next(it.islice(it.combinations_with_replacement(range(10080), 2), b))[0]])
    print "player_one_template is \n", player_one_template
    player_two_util_possibilities = [] #initiate empty. This should end up being a list of 36 elements
    player_two_template_num = next(it.islice(it.combinations_with_replacement(range(10080), 2),b))[1]
    print "player_two_template_num = ", player_two_template_num

    print "finding player_switch_templates that equal ", player_two_template_num, " ..."
    for i, x in enumerate(player_switch_templates):
        if x==player_two_template_num:
            player_two_util_possibilities.append(i)
    print "... finished. Found ", len(player_two_util_possibilities), " permutations that correspond to this player_two_template_num"
    print "player_two_util_possibilities = ", player_two_util_possibilities
    output = [] # initialize empty

    for i in range(len(player_two_util_possibilities)):
        print "i = ", i
        print "player_two_util_possibilities[i] = ", player_two_util_possibilities[i]
        slice_var = it.islice(it.permutations(range(1,10)), player_two_util_possibilities[i], None)
        print "slice_var = ", slice_var
        player_two_utils = list(next(slice_var, None))
        print "player_two_utils = ", player_two_utils
        player_two_template = np.array(player_two_utils).reshape(3, 3)
        print "player_two_template = ", player_two_template
        print "appending: \n", np.array([player_one_template, player_two_template]), "\n"
        output.append(np.array([player_one_template, player_two_template]))

    return output

def main():
    global player_switch_templates
    global templates

    import itertools as it
    import numpy as np
    import pickle
    import os.path

    fileObject = open('player_switch_templates.p','r')
    print "loading file"
    player_switch_templates = pickle.load(fileObject)
    fileObject.close()

    fileObject = open('templates.p','r')
    print "loading file"
    templates = pickle.load(fileObject)
    fileObject.close()

    #now the kth element (game) in the bth bucket is given by the template:
    #templates[bucketList[b][0][0]] 
    #and player 2's utilities filled in with the permutation:
    #list(list(it.permutations(range(1,10)))[bucketList[b][1][k]])

# boilerplate
if __name__ == '__main__':
  main()
