import sys
import pandas as pd
import numpy as np
from sklearn import cluster
from Same_Size_K_Means.clustering.equal_groups import EqualGroupsKMeans

df_data = pd.read_csv('data.tsv', sep='\t')
df_rushee_names = pd.read_csv('rushees.tsv', sep='\t')
rooms = 2
rushees_per_room = {0: 2, 1: 2}

# Builds our data matrix with feature vectors v where v[i]
# is 1 if that member has met rushee i, 0 if that member has not
# met rushee i.
#
# Also builds our vector of member names m, rushee names r, where
# i is the index used to represent the member at m[i] and r[i].
def getData():
	matrix = []
	member_names = []
	rushee_names = []
	num_rushees = len(df_rushee_names.index)
	for index, row in df_rushee_names.iterrows():
		rushee_names.append(row[0])
	for index, row in df_data.iterrows():
		member_names.append(row[1])
		rushees_met = row[2]
		feature_vec = np.zeros(num_rushees)
		for rushee in rushees_met.split(', '):
			idx = rushee_names.index(rushee)
			feature_vec[idx] = 1
		matrix.append(feature_vec)
	return matrix, member_names, rushee_names, num_rushees

matrix, member_names, rushee_names, num_rushees = getData()

# Provides all possible room assigments given the number of individuals
# to be placed per room and total number of individuals
def gra(curArr, curIndex, allArrs):
	if (curIndex == num_rushees):
		newArr = curArr.copy()
		allArrs.append(newArr)
	for key, value in rushees_per_room.items():
		if (value > 0):
			curArr[curIndex] = key
			rushees_per_room[key] = value - 1
			gra(curArr, curIndex + 1, allArrs)
			rushees_per_room[key] = value

# Calculate loss provided some room assignments
def loss(rushee_room_assignments, member_room_assignments, matrix):
	loss = 0
	for rushee_idx, rushee_room in enumerate(rushee_room_assignments):
		for member_idx, member_room in enumerate(member_room_assignments):
			if (rushee_room == member_room):
				if (matrix[member_idx][rushee_idx] == 1):
					loss += 1
	return loss

# Prints out the current room assignments
def print_room(rushee_room_assignments, member_room_assignments):
	for room_num in range(rooms):
		print('\n==== Room Number:',room_num + 1,'====')
		print('--- RUSHEES:')
		for rushee_idx, rushee_room in enumerate(rushee_room_assignments):
			if (rushee_room == room_num):
				print('•', rushee_names[rushee_idx])
		print('--- MEMBERS:')
		for member_idx, member_room in enumerate(member_room_assignments):
			if (member_room == room_num):
				print('•', member_names[member_idx])
		print('========================\n')

# Determines the optimal room assignment
def best_assignment(total_assignments, member_room_assignments):
	min_loss = sys.maxsize
	min_idx = 0
	for idx, rushee_room_assignments in enumerate(total_assignments):
		cur_loss = loss(rushee_room_assignments, member_room_assignments, matrix)
		if (cur_loss < min_loss):
			min_idx = idx
			min_loss = cur_loss
	print(f'Optimal Room Assignment (Loss: {min_loss})')
	print_room(total_assignments[min_idx], member_room_assignments)

# K means needs to be same size clusters
k_means = cluster.KMeans(n_clusters=rooms)
k_means.fit(matrix)

clf = EqualGroupsKMeans(n_clusters=rooms)
clf.fit(matrix)
# print(k_means.labels_)
# print(clf.labels_)
total_assignments = []
gra(np.zeros(num_rushees), 0, total_assignments)

print('--- best_assignment() calculated using k_means ---')
best_assignment(total_assignments, k_means.labels_)
print('--- best_assignment() calculated using EqualGroupsKMeans ---')
best_assignment(total_assignments, clf.labels_)
