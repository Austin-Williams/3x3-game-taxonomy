#The purpose of this program is to generate a list of all 10,080 'templates', then pickle it as 'templates.p'

from itertools import combinations, permutations
from numpy import matrix
import pickle


# For refrence, here is a 3x3 'matrix with entries from {1,9}x{1,9}'
#-------------------------
#| (a,b) | (c,d) | (e,f) |
#-------------------------
#| (g,h) | (i,j) | (k,l) |
#-------------------------
#| (m,n) | (o,p) | (q,r) |
#-------------------------

#Note that for canonical form matrices, a = 9. Moreover 4 <= i < 9, and k,o,q are chosen from the set [1, i-1].

#Templates have the following form:

#-------------------------
#| (a,*) | (c,*) | (e,*) |
#-------------------------
#| (g,*) | (i,*) | (k,*) |
#-------------------------
#| (m,*) | (o,*) | (q,*) |
#-------------------------

#We'll store templates in templates.p as matrices of the form:

#-------------
#| a | c | e |
#-------------
#| g | i | k |
#-------------
#| m | o | q |
#-------------

#Again, a=9, 4 <= i < 9, and k,o,q are chosen from the set [1, i-1].

def main():

	output_list=[[]]*10080 #This is a list that we'll populate with the template matrices as we produce them.
	t = 0 #This is the variable we'll use to keep track of how many matrices we've stored so far.

	a = 9

	for i in range(4,9):

		#i is the ordinal utility that player 1 is assigning to the outcome represented by the middle square 
		#in the 3x3 matrix (labeled as i in the comments at the beggining of ). Once i is chosen, we know the
		#entries k, o, and q must be chosen from the set, refered to as 'minor_set' below, of the numbers [1, i-1].
		minor_set = range(1,i)

		#Next we list all possible ways of choosing 3 elements from minor_set.
		possible_minor_choices = list(combinations(minor_set, 3))

		for choice in possible_minor_choices:
			#for each of these, we look at each possible permutation of the three elements.
			minor_perms = list(permutations(choice))

			for minor_perm in minor_perms:

				[k,o,q] = minor_perm #we assign utility values to k, o, and q.

				#now that a,i,k,o,and q have been determined, we look at the remaining unassigned utilities. The 
				#set of remaining unassigned utilities is referred to as 'major_set' below. Note that major_set
				#contains exactly 4 values, and these values will be assigned to c,e,g, and m in some order.
				major_set = list(set(range(1,9))-set(minor_perm)-set([i]))

				#now we consider each possible way of assigning the four values in major_set to the variables c,e,g and m.
				major_perms = list(permutations(major_set))

				for major_perm in major_perms:
					[c,e,g,m] = major_perm
					output_list[t] = matrix([[a,c,e],[g,i,k],[m,o,q]])
					t = t+1

	#when this line of code is reached, all the matrices have been stored in output_list. We need only store the results.
	fileObject = open("templates.p", "wb")
	pickle.dump(output_list, fileObject)
	fileObject.close()

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  main()
