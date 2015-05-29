#The purpose of this program is to create a 362,880 element long list that shows the correspondence between
#all permutations of range(1,10) and the templates to which the correspond. Given a permutation of range(1,10), we
#find the corresponding template as follows:
#Enter the values of the permutaion into a matrix (think of this as entering in payer 2's utilities into a matrix and
#leaving player 1's blank). Then take the transposition of the matrix (you now have something that looks like a template
#but likely isn't in 'standard form'). Next, put the matrix into standard form, and you're done. The result is the template
#that corresponds to the permutation.
#
#Once all of these are stored, we'll pickle the result as perm_templates.p for later use. The output will be a list where
#the the ith element of the list is the template that corresponds to the ith permutaion in list(permutations(range(1,10))).
#The pickled file is about 140mb.

from itertools import combinations, permutations
from numpy import *
import pickle

def standard_form(mat):
	#This function takes a matrix and exchanges rows/cols as needed to put the matrix
	#into 'standard form', so that '9' appears in the upper left hand corner, and the
	#largest value in the lower right 2x2 submatrix appears in the center of the input matrix.
	#The function returns the matrix in this standard form.

	#we find the index of the highest valued entry in the matrix (9 in our case).
	index_of_max = argmax(mat)
	indices = unravel_index(index_of_max,[3,3], order='C')

	#Then we swap rows/cols to put that highest valued entry in the upper left hand corner of the matrix.
	mat[[0,indices[0]],:] = mat[[indices[0],0],:]
	mat[:,[0,indices[1]]] = mat[:,[indices[1],0]]

	#Next we look at the lower right 2x2 submatrix and do the same thing.
	submatrix = mat[1:,1:]
	index_of_submax = argmax(submatrix)
	subindices = unravel_index(index_of_submax,[2,2], order='C')

	#we swap rows/cols as needed.
	mat[[1,subindices[0]+1],:] = mat[[subindices[0]+1,1],:]
	mat[:,[1,subindices[1]+1]] = mat[:,[subindices[1]+1,1]]

	return mat
#---End of standard_form def.

def perm_template(perm):
	#This function takes in a permutation of range(1,10) (as a list). It converts the permutation to a matrix,
	#takes the transposition of the matrix, puts the matrix into standard form, and then outputs the standard form
	#matrix.
	mat = matrix([[perm[0], perm[1], perm[2]],[perm[3], perm[4], perm[5]],[perm[6], perm[7], perm[8]]])
	mat = mat.transpose()
	mat = standard_form(mat)

	return mat
#---End of per_template def.

def main():
	i = 0 #this will keep track of how many permutaion/template pairs we've already stored
	output_list = [[]]*362880 #This is
	perms = list(permutations(range(1,10)))
	for perm in perms:
		output_list[i]=[perm_template(perm)]
		i=i+1
		if i%1000 == 0:
			print i*100/362880, " percent complete." #it's nice to see progress displayed.

	#when this line of code is reached, all the permutation/template pairs have been stored in output_list. 
	#We need only store the results.
	print 'Storing templates.'
	fileObject = open("perm_templates.p", "wb")
	pickle.dump(output_list, fileObject)
	fileObject.close()


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  main()
