import os
import pandas as pd
import matplotlib.pyplot as plt


def return_file_line(fname):
    
    reader = open(fname, "r")
    lines = reader.readlines()
    
    #print(lines[:200])
    return lines

def return_file_axis(lines):
    
    x = []
    yi = []
    yk = []
    yt = []
    for line in lines:
        if(line.find('tid') != -1):
            fields = line.split('|')
            tidkj = fields[0]
            iend = fields[1]
            wave = int(fields[3].split(':')[1].replace(' ', ''))
            t = fields[5]

            tid = int(tidkj.split('-')[0].split(':')[1].replace(' ', ''))
            k = int(tidkj.split('-')[1].split(':')[1].replace(' ', ''))
            i = int(iend.split('/')[0].split(':')[1].replace(' ', ''))
            t = int(t.split(':')[1].strip('\n').replace(' ', ''))

            #print(line)
            #print(tid, k, i, t)
            #exit(1)
            
            x.append(32*wave+tid)
            yk.append(k)
            yi.append(i)
            yt.append(t)
            
            


    return x, yi, yk, yt


def one_plot(df, name):

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('warp*wave')
    ax1.set_ylabel('mat[i]', color=color)
    ax1.plot(df['warp*wave'], df['matrix index'], color = color, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)

    ax1.set_title(name)
    
    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('changed bit', color=color)
    ax2.plot(df['warp*wave'], df['changed bit'], color=color, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)

    
    
if __name__ == "__main__":


    file_names = ['plain_beg_36.txt','plain_mid_36.txt','plain_end_36.txt',
                  'plain_beg_37.txt','plain_mid_37.txt','trans_end_37.txt']
                  
        
    for name in file_names:
    
        lines = return_file_line(name)
        x, yi, yk, yt = return_file_axis(lines)
        df = pd.DataFrame({'warp*wave': x,
                           'iter': yi,
                           'changed bit': yk,
                           'matrix index': yt})
        
        one_plot(df, name)
        
    plt.tight_layout()
    plt.show()
    
    
    
    
