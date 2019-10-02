# Interview Room Matcher

_Note: You must clone the [Same Size K Means repository](https://github.com/ndanielsen/Same-Size-K-Means) in the root directory of this repository and rename the file from `Same-Size-K-Means` to `Same_Size_K_Means` in order to use the same size k means clustering feature of this program._

## Usage
Run ```python matcher.py``` in the root directory with your files. Your file with all the matches must be titled `data.tsv` and the list of rushees must be titled `rushees.tsv`.

## Details
_Abstract:_ Throughout a networking event, members will get to meet a large number of candidates. Following this networking event, interviews are conducted to evaluate the candidates. When it comes to interviews, we don't want members to see the same faces that they have already networked with. To increase each candidate's exposure, we want to create room assignments that maximize the number of new faces shown to interviewers. This allows more individuals to provide feedback on a candidate when discussing qualifications.

_Problem:_ How can we create interview rooms such that we maximize the number of new faces shown given the different candidates each member has already been exposed to?

_Solution:_ We tackle this problem by breaking it into two pieces. First, finding the optimal way to group members up. Then, finding the optimal room assignments per candidate. To achieve this, the program takes the following steps:
1. Two arrays are built, where each array maps the member/rushee's name to the index.
2. A feature matrix is built, where the (i, j) entry is 1 if member i has interacted with candidate j and 0 if not.
3. The K Means algorithm is used to cluster our members into groups based on which candidates they have interacted with. These clusters will be the rooms that members are placed into. In this case, equal size k means will be more optimal because we want each interview room to contain roughly the same number of members.
4. An exhaustive search of every candidate-to-room combination is attempted, and for each combination, a loss value is determined. The loss function calculates the number of repeated interactions are occuring per room. For each member that is encountering a candidate that they have already interacted with, we increase our loss by 1.
5. The optimal rooms and assignments are determined by selecting the candidate-to-room combination that minimizes our loss function.
