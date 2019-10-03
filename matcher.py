import sys
import pandas as pd
import numpy as np
from sklearn import cluster
from Same_Size_K_Means.clustering.equal_groups import EqualGroupsKMeans

# Builds our data matrix with feature vectors v where v[i]
# is 1 if that member has met rushee i, 0 if that member has not
# met rushee i.
#
# Also builds our vector of member names m, rushee names r, where
# i is the index used to represent the member at m[i] and r[i].
def getDataMatrix(df_data):
	matrix = []
	member_names = []
	rushee_names_all = []

	# Build rushee and member arrays
	for index, row in df_data.iterrows():
		member_names.append(row[1])
		rushees_met = row[2]
		for rushee in rushees_met.split(', '):
			if (rushee not in rushee_names_all):
				rushee_names_all.append(rushee)
	total_rushees = len(rushee_names_all)
	# Build feature matrix
	for index, row in df_data.iterrows():
		rushees_met = row[2]
		feature_vec = np.zeros(total_rushees)
		for rushee in rushees_met.split(', '):
			idx = rushee_names_all.index(rushee)
			feature_vec[idx] = 1
		matrix.append(feature_vec)
	return matrix, member_names, rushee_names_all

# Provides all possible room assigments given the number of individuals
# to be placed per room and total number of individuals
def gra(curArr, curIndex, allArrs, rushees_per_room, num_rushees):
	if (curIndex == num_rushees):
		newArr = curArr.copy()
		allArrs.append(newArr)
	for key, value in rushees_per_room.items():
		if (value > 0):
			curArr[curIndex] = key
			rushees_per_room[key] = value - 1
			gra(curArr, curIndex + 1, allArrs, rushees_per_room, num_rushees)
			rushees_per_room[key] = value

# Calculate loss provided some room assignments
def loss(rushee_room_assignments, member_room_assignments, matrix, rushee_names_all, rushee_names_cur):
	loss = 0
	for rushee_idx, rushee_room in enumerate(rushee_room_assignments):
		name = rushee_names_cur[rushee_idx]
		rushee_idx = rushee_names_all.index(name)
		for member_idx, member_room in enumerate(member_room_assignments):
			if (rushee_room == member_room):
				if (matrix[member_idx][rushee_idx] == 1):
					loss += 1
	return loss

# Prints out the room assignments for the members
def print_members_rooms(member_room_assignments, member_names, rooms):
	for room_num in range(rooms):
		print('\n==== Room Number:',room_num + 1,'====')
		for member_idx, member_room in enumerate(member_room_assignments):
			if (member_room == room_num):
				print('•', member_names[member_idx])
		print('========================\n')


# Determines the optimal room assignment
def best_assignment(matrix, total_assignments, member_room_assignments, rushee_names_all, rushee_names_cur):
	min_loss = sys.maxsize
	min_idx = 0
	for idx, rushee_room_assignments in enumerate(total_assignments):
		cur_loss = loss(rushee_room_assignments, member_room_assignments, matrix, rushee_names_all, rushee_names_cur)
		if (cur_loss < min_loss):
			min_idx = idx
			min_loss = cur_loss
	return total_assignments[min_idx], min_loss

# Prompts user to input information
def main():
	print('Welcome to Interview Room Matcher.')
	print('What is the name/directory of your interactions file relative to current directory?')
	member_interactions_file_name = input("File (must be .tsv): ")
	rooms = int(input("How many interview rooms are there? "))

	# Read data and make clusters for members
	df_data = pd.read_csv(member_interactions_file_name, sep='\t')
	matrix, member_names, rushee_names_all = getDataMatrix(df_data)
	clf = EqualGroupsKMeans(n_clusters=rooms)
	clf.fit(matrix)
	member_room_assignments = clf.labels_
	print_members_rooms(member_room_assignments, member_names, rooms)

	while(True):
		print("What is the name/directory of your rushee list relative to current directory?")
		rushees_file_name = input("File (should be .txt with new lines): ")
		df_rushee_names_cur = pd.read_csv(rushees_file_name)
		rushee_names_cur = []
		for index, row in df_rushee_names_cur.iterrows():
			rushee_names_cur.append(row[0])

		# Determine number of rushees per room
		num_rushees = len(df_rushee_names_cur.index)
		rushees_per_room = {}
		for i in range(rooms):
			rushees_per_room[i] = num_rushees // rooms;
		for i in range(num_rushees % rooms):
			rushees_per_room[i] += 1

		# Find optimal room assignment
		total_assignments = []
		gra(np.zeros(num_rushees), 0, total_assignments, rushees_per_room, num_rushees)
		best_rushee_room_assignment, min_loss = best_assignment(matrix, total_assignments,
			member_room_assignments, rushee_names_all, rushee_names_cur)

		# Print our findings
		print(f'Optimal Room Assignment (Loss: {min_loss})')
		for room_num in range(rooms):
			print('\n==== Room Number:',room_num + 1,'====')
			for rushee_idx, rushee_room in enumerate(best_rushee_room_assignment):
				if (rushee_room == room_num):
					print('•', rushee_names_cur[rushee_idx])
			print('========================\n')

main()
