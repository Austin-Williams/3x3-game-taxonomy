#The purpose of this program is to create a 362,880 element long list that shows the correspondence between
#all permutations of range(1,10) and the templates to which the correspond. Given a permutation of range(1,10), we
#find the corresponding template as follows:
#Enter the values of the permutaion into a matrix (think of this as entering in payer 2's utilities into a matrix and
#leaving player 1's blank). Then take the transposition of the matrix (you now have something that looks like a template
#but likely isn't in 'standard form'). Next, put the matrix into standard form, and you're done. The result is the template
#that corresponds to the permutation.
#
#Once all of these are stored, we'll pickle the result as player_switch_templates.p for later use. The output will be a list
#called player_switch_templates such that: when player_switch_template[i]=j, the template that corresponds to the ith 
#permutaion in  list(permutations(range(1,10))) is the template found in templates[j]. The pickled file is about 2.5mb.

from itertools import combinations, permutations
import numpy as np
import pickle
import os.path
import hashlib

def standard_form(mat):
    #This function takes a matrix and exchanges rows/cols as needed to put the matrix
    #into 'standard form', so that '9' appears in the upper left hand corner, and the
    #largest value in the lower right 2x2 submatrix appears in the center of the input matrix.
    #The function returns the matrix in this standard form.

    #we find the index of the highest valued entry in the matrix (9 in our case).
    index_of_max = np.argmax(mat)
    indices = np.unravel_index(index_of_max,[3,3], order='C')

    #Then we swap rows/cols to put that highest valued entry in the upper left hand corner of the matrix.
    mat[[0,indices[0]],:] = mat[[indices[0],0],:]
    mat[:,[0,indices[1]]] = mat[:,[indices[1],0]]

    #Next we look at the lower right 2x2 submatrix and do the same thing.
    submatrix = mat[1:,1:]
    index_of_submax = np.argmax(submatrix)
    subindices = np.unravel_index(index_of_submax,[2,2], order='C')

    #we swap rows/cols as needed.
    mat[[1,subindices[0]+1],:] = mat[[subindices[0]+1,1],:]
    mat[:,[1,subindices[1]+1]] = mat[:,[subindices[1]+1,1]]

    return mat
#---End of standard_form def.

def perm_template(perm):
    #This function takes in a permutation of range(1,10) (as a list). It converts the permutation to a matrix,
    #takes the transposition of the np.matrix, puts the matrix into standard form, and then outputs the standard form
    #matrix.
    mat = np.matrix([[perm[0], perm[1], perm[2]],[perm[3], perm[4], perm[5]],[perm[6], perm[7], perm[8]]])
    mat = mat.transpose()
    mat = standard_form(mat)
    mat = mat.copy(order='C')

    return mat
#---End of per_template def.

def initialize():
    #we'll need to access 'templateHashes.p', so first we check that we have access to it. If not, then we create it.
    if not os.path.isfile('templateHashes.p'):
        print "The file 'templateHashes.p' does not exist. Creating it now. This may take a few seconds."
        import create_templates
        create_templates.main()
        print "The file 'templateHashes.p' has been created."

    #We load the templates from temlates.p to the template variable.
    fileObject = open('templateHashes.p','r')
    print "loading file"
    templateHashes = pickle.load(fileObject)
    fileObject.close()

    return templateHashes

def main():

    templateHashes = initialize()

    i = 0 #this will keep track of how many template-indices we've already stored
    output_list = [[]]*362880 #This is an initialized list we'll use to store the templates
    perms = list(permutations(range(1,10)))
    for perm in perms:
        #The next line of code does 4 things. It calculates the player switch template: [perm_template(perm)].
        #Then is hashes the result: hashlib.md5(*).hexdigest().
        #Then it looks up what template number we just found: templateHashes.index(*).
        #Then it stores the result in output_list[i].

        output_list[i] = templateHashes.index( hashlib.md5(perm_template(perm)).hexdigest() )

        i=i+1
        if i%1000 == 0:
            print i*100/362880, " percent complete." #it's nice to see progress displayed.

    #when this line of code is reached, all the template indices have been stored in output_list. 
    #We need only pickle the results.
    print 'Storing templates.'
    fileObject = open("player_switch_templates.p", "wb")
    pickle.dump(output_list, fileObject)
    fileObject.close()


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  main()
