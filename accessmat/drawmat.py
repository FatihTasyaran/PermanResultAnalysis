import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def return_matrix(filename):

    reader = open(filename , "r")
    lines = reader.readlines()

    print(len(lines))

    mat = {}

    total = 0
    
    for i in range(len(lines)):

        mat[str(i)] = []

        linestrip = lines[i].strip("\n")
        values = linestrip.split(" ")

        for val in values:
            try:
                mat[str(i)].append(int(val) + 1)
                total += int(val) + 1
            except:
                x = 1
        

    #for key in mat:
        #print(mat[key])
        
    df = pd.DataFrame.from_dict(mat, orient='index')

    return df, total


def draw_matrix(df):

    sns.heatmap(df)
    plt.show()

if __name__ == "__main__":

    mat, total = return_matrix("transpose32_2.txt")
    draw_matrix(mat)
    print("Transposed total access: ", '{:.2e}'.format(total))

    mat2, total2 = return_matrix("plain32_2.txt")
    draw_matrix(mat2)
    print("Plain total access: ", '{:.2e}'.format(total2))
    
