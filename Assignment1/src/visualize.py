import matplotlib.pyplot as plt
import numpy as np
import csv 

filename_vectorAdd = '../build/vectorAddition.csv'
filename_matrixRot = '../build/matrixRot.csv'

VECTOR_SIZE_INDEX_IN_CSV_FILE = 3
WORK_GROUP_DX_INDEX_IN_CSV_FILE = 0
WORK_GROUP_DY_INDEX_IN_CSV_FILE = 1
NORMAL_TIMING_INDEX_IN_CSV_FILE = 4
OPT_TIMING_INDEX_IN_CSV_FILE = 5

# ----- UTIL FUNCTIONS FOR DEALING WITH CSV FILES -----
def read_csv_file(name, fields, rows):

    with open(name, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile, dialect='excel', delimiter=";")
        # extracting field names through first row
        fields = next(csvreader)
        for row in csvreader:
            # print(', '.join(row))
            rows.append(row)
        return fields, rows

def take_timing(elem):
    return elem[4]

def worksize_grp(elem):
    return float(elem[0])

def worksize_grp_two_dim(elem):
    return float(elem[0]) * float(elem[1])

def read_in_row_groups_fromm_different_vector_sizes(rows, rows_with_different_vecsizes, vector_sizes):
    for i in range(len(vector_sizes)):
        rows_with_different_vecsizes.append([row for row in rows if int(row[VECTOR_SIZE_INDEX_IN_CSV_FILE]) == vector_sizes[i]])
    return rows_with_different_vecsizes

def calc_scale_of_scatter_points_according_to_elapsed_time(scale, rows_with_different_vecsizes):

    for i in range(len(rows_with_different_vecsizes)):
        # some scaling magic... playing with linear and scaling factor ...
        scaling = 3
        linear_offset = 0.0
        scale.append([((float(row[4])+linear_offset) * scaling)**2 for row in rows_with_different_vecsizes[i]])

    return scale

def scale_points_for_matrix_rotation_analysis(scale, time_vec):

    for i in range(len(time_vec)):
        # some scaling magic... playing with linear and scaling factor ...
        scaling = 0.5
        linear_offset = 0
        scale.append([((float(tmp[0]) + linear_offset) * scaling)**2 for tmp in time_vec[i]])

    return scale    

# ----- VECTOR ADDITION STUFF -----
# reading all infos in a order way
fields = []
rows = []
vector_sizes = [4194304, 1048576, 262144, 65536, 16384, 4096]

read_csv_file(filename_vectorAdd, fields, rows)

rows_with_different_vecsizes = []
read_in_row_groups_fromm_different_vector_sizes(rows, rows_with_different_vecsizes, vector_sizes)

dx_from_different_vecsizes = []
timing_from_different_vecsizes = []


for row_groups in rows_with_different_vecsizes:
    # we want the lowest value on the very left
    row_groups.sort(key=worksize_grp)

for rows_one_vecsize in rows_with_different_vecsizes:
    tmp_dx_arr = []
    tmp_timing_array = []
    for row in rows_one_vecsize:
        tmp_dx_arr.append([float(row[WORK_GROUP_DX_INDEX_IN_CSV_FILE])])
        tmp_timing_array.append([float(row[NORMAL_TIMING_INDEX_IN_CSV_FILE])])
    dx_from_different_vecsizes.append(tmp_dx_arr)
    timing_from_different_vecsizes.append(tmp_timing_array)


scale = []
calc_scale_of_scatter_points_according_to_elapsed_time(scale, rows_with_different_vecsizes)

# scale_vecsize_1048576 = [float(float(row[4]))**2 + 10 for row in rows_with_vecsize_1048576]
# scale_vecsize_262144 = [float(float(row[4]))**2 + 10 for row in rows_with_vecsize_262144]
plt.scatter(dx_from_different_vecsizes[0], timing_from_different_vecsizes[0], s=scale[0], c='magenta', label="4194304")
plt.scatter(dx_from_different_vecsizes[1], timing_from_different_vecsizes[1], s=scale[1], c='green', label="1048576")
plt.scatter(dx_from_different_vecsizes[2], timing_from_different_vecsizes[2], s=scale[2], c='grey', label="262144")
plt.scatter(dx_from_different_vecsizes[3], timing_from_different_vecsizes[3], s=scale[3], c='blue', label="65536")
plt.plot([], [], ' ', label="Vector sizes")
plt.title("Vector Addition")
plt.xlabel('Worksizegroup dx')
plt.ylabel('timing [ms]')
plt.grid(True, linestyle='-.')
plt.legend(loc=0, fontsize='x-small')
plt.savefig("VectorAddition.png")
plt.show()
plt.clf()

# -----MATRIX ROTATION STUFF 
# reading all infos in a order way
fields = []
rows = []
vector_sizes = [40960000, 25600000, 20480000, 12800000, 10240000]

read_csv_file(filename_matrixRot, fields, rows)

rows_with_different_vecsizes = []
read_in_row_groups_fromm_different_vector_sizes(rows, rows_with_different_vecsizes, vector_sizes)
        
workgroupsize_from_different_vecsizes = []
timing_from_different_vecsizes = []
opt_timing_from_different_vecsizes = []

for row_groups in rows_with_different_vecsizes:
    # we want the lowest value on the very left
    row_groups.sort(key=worksize_grp_two_dim)

for rows_one_vecsize in rows_with_different_vecsizes:
    tmp_workgroupsize_arr = []
    tmp_timing_array = []
    tmp_opt_timing_array = []

    for row in rows_one_vecsize:
        # we have this way more workgroup sizes than timing values for symmetric dy, dx axis workgroup size
        # if one workgroup size already consists average the result !
        if [float(row[WORK_GROUP_DX_INDEX_IN_CSV_FILE]) * float(row[WORK_GROUP_DY_INDEX_IN_CSV_FILE])] in tmp_workgroupsize_arr:
            index = tmp_workgroupsize_arr.index([float(row[WORK_GROUP_DX_INDEX_IN_CSV_FILE]) * float(row[WORK_GROUP_DY_INDEX_IN_CSV_FILE])])
            tmp_timing_val = tmp_timing_array[index][0]
            tmp_timing_array[index] = [(tmp_timing_val + float(row[NORMAL_TIMING_INDEX_IN_CSV_FILE]))/2]
            tmp_opt_timing_val = tmp_opt_timing_array[index][0]
            tmp_opt_timing_array[index] = [(tmp_opt_timing_val + float(row[NORMAL_TIMING_INDEX_IN_CSV_FILE]))/2]

        # if workgroupsize for symmetry reasons do not exist make new entry
        else:
            tmp_workgroupsize_arr.append([float(row[WORK_GROUP_DX_INDEX_IN_CSV_FILE]) * float(row[WORK_GROUP_DY_INDEX_IN_CSV_FILE])])
            tmp_timing_array.append([float(row[NORMAL_TIMING_INDEX_IN_CSV_FILE])])
            tmp_opt_timing_array.append([float(row[OPT_TIMING_INDEX_IN_CSV_FILE])])

    workgroupsize_from_different_vecsizes.append(tmp_workgroupsize_arr)
    timing_from_different_vecsizes.append(tmp_timing_array)
    opt_timing_from_different_vecsizes.append(tmp_opt_timing_array)

scale_naive = []
scale_optimized = []

# for rows in rows_with_different_vecsizes:

#     for row in rows:
#         # IMPORTANT NOTE: 2nd and third entry are obsolete
#         # but i keep them for reusability of function created originally for vector addition
#         naive_entry = [[float(row[0]) * float(row[1])], row[1], row[2], row[3], row[4]]
#         optimized_entry = [[float(row[0][0]) * float(row[1][0])], row[1], row[2], row[3], row[5]]

#         # find out whether there is already an entry with this wrk group size
#         if ((row[0] for row in rows_with_combined_wrk_group_size_naive) is naive_entry[0]):
#             row[4] = (row[4] + naive_entry[4]) / 2
#         else:
#             rows_with_combined_wrk_group_size_naive.append(naive_entry)

#         if ((row[0] for row in rows_with_combined_wrk_group_size_optimized) is optimized_entry[0]):
#             row[4] = (row[4] + optimized_entry[4]) / 2
#         else:
#             rows_with_combined_wrk_group_size_optimized.append(optimized_entry)

scale_points_for_matrix_rotation_analysis(scale_naive, timing_from_different_vecsizes)
scale_points_for_matrix_rotation_analysis(scale_optimized, opt_timing_from_different_vecsizes)

# unoptimized algorithm
plt.scatter(workgroupsize_from_different_vecsizes[0], timing_from_different_vecsizes[0] , s=scale_naive[0], c='magenta', label="40960000 (naive)") 
plt.scatter(workgroupsize_from_different_vecsizes[1], timing_from_different_vecsizes[1], s=scale_naive[1], c='green', label="25600000 (naive)")
plt.scatter(workgroupsize_from_different_vecsizes[2], timing_from_different_vecsizes[2], s=scale_naive[2], c='grey', label="20480000 (naive)")
plt.scatter(workgroupsize_from_different_vecsizes[3], timing_from_different_vecsizes[3], s=scale_naive[3], c='blue', label="12800000 (naive)")

# optimized algorithm
plt.scatter(workgroupsize_from_different_vecsizes[0], opt_timing_from_different_vecsizes[0], s=scale_optimized[0], c='magenta', label="40960000 (optimized)", marker="^")
plt.scatter(workgroupsize_from_different_vecsizes[1], opt_timing_from_different_vecsizes[1], s=scale_optimized[1], c='green', label="25600000 (optimized)", marker="^")
plt.scatter(workgroupsize_from_different_vecsizes[2], opt_timing_from_different_vecsizes[2], s=scale_optimized[2], c='grey', label="20480000 (optimized)", marker="^")
plt.scatter(workgroupsize_from_different_vecsizes[3], opt_timing_from_different_vecsizes[3], s=scale_optimized[3], c='blue', label="12800000 (optimized)", marker="^")

plt.plot([], [], ' ', label="Vector sizes")
plt.title("Matrix Rotation")
plt.xlabel('Worksizegroup')
plt.ylabel('timing [ms]')
plt.grid(True, linestyle='-.')
plt.legend(loc=0, fontsize='x-small')
plt.savefig("MatrixRotation.png")
plt.show()