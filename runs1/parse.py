import pandas as pd
import glob
import statistics
import matplotlib.pyplot as plt

keys = ['cpu',
        'gpu',
        'sparse',
        'dense',
        'exact',
        'approximation',
        'halfprec',
        'quadprec',
        'halfstore',
        'quadstore',
        'pattern',
        'isgrid',
        'gridm',
        'gridn',
        'algo',
        'threads',
        'sinterval',
        'stime',
        'fname',
        'type',
        'no_rep',
        'ordering',
        'gpu_num',
        'no_times',
        'approx_act_no_times',
        'grid_dim',
        'block_dim',
        'device_id',
        'grid_multip',
        'decomposition',
        'scaling_thresh',
        'host',
        'algo_name',
        'command',
        'time',
        'acc_result_avg',
        'acc_max',
        'acc_min',
        'acc_stdev',
        'acc_err',
        'synth_size',
        'synth_density',
        'synth_type',
        'realw_name']


def get_files():

    files = glob.glob("*.stdtxt")
    return files

def common_fields(run_dict, lines):

    ##command
    command = lines[0][lines[0].find(':') + 2:]
    run_dict['command'] = command

    ##cpu
    cpu = lines[2][lines[2].find(':') + 2:]
    run_dict['cpu'] = int(cpu)

    ##gpu
    gpu = lines[3][lines[3].find(':') + 2:]
    run_dict['gpu'] = int(gpu)

    ##sparse
    sparse = lines[4][lines[4].find(':') + 2:]
    run_dict['sparse'] = int(sparse)

    ##dense
    dense = lines[5][lines[5].find(':') + 2:]
    run_dict['dense'] = int(dense)

    ##exact
    exact = lines[6][lines[6].find(':') + 2:]
    run_dict['exact'] = int(exact)

    #approximation
    approximation = lines[7][lines[7].find(':') + 2:]
    run_dict['approximation'] = int(approximation)

    ##halfprec
    halfprec = lines[8][lines[8].find(':') + 2:]
    run_dict['halfprec'] = int(halfprec)

    ##quadprec
    quadprec = lines[9][lines[9].find(':') + 2:]
    run_dict['quadprec'] = int(quadprec)

    ##halfstore
    halfstore = lines[10][lines[10].find(':') + 2:]
    run_dict['halfstore'] = int(halfstore)

    ##quadstore
    quadstore = lines[11][lines[11].find(':') + 2:]
    run_dict['quadstore'] = int(quadstore)
    
    ##pattern
    pattern = lines[12][lines[12].find(':') + 2:]
    run_dict['pattern'] = int(pattern)

    ##isgrid
    isgrid = lines[13][lines[13].find(':') + 2:]
    run_dict['isgrid'] = int(isgrid)

    ##gridm
    gridm = lines[14][lines[14].find(':') + 2:]
    run_dict['gridm'] = int(gridm)

    ##gridn
    gridn = lines[15][lines[15].find(':') + 2:]
    run_dict['gridn'] = int(gridn)

    ##algo
    algo = lines[16][lines[16].find(':') + 2:]
    run_dict['algo'] = int(algo)

    ##threads
    threads = lines[17][lines[17].find(':') + 2:]
    run_dict['threads'] = int(threads)

    ##sinterval
    sinterval = lines[18][lines[18].find(':') + 2:]
    run_dict['sinterval'] = int(sinterval)

    ##stime
    stime = lines[19][lines[19].find(':') + 2:]
    run_dict['stime'] = int(stime)

    ##fname
    fname = lines[20][lines[20].find(':') + 2:]
    fname = fname[fname.rfind('/') + 1:].replace(' ', '')
    run_dict['fname'] = fname

    ##ttype
    ttype = lines[21][lines[21].find(':') + 2:]
    run_dict['type'] = ttype

    ##no_rep
    no_rep = lines[22][lines[22].find(':') + 2:]
    run_dict['no_rep'] = int(no_rep)
    
    ##ordering
    ordering = lines[23][lines[23].find(':') + 2:]
    run_dict['ordering'] = int(ordering)

    ##gpu_num
    gpu_num = lines[24][lines[24].find(':') + 2:]
    run_dict['gpu_num'] = int(gpu_num)

    ##number_of_times
    number_of_times = lines[25][lines[25].find(':') + 2:]
    run_dict['no_times'] = int(number_of_times)
    
    ##grid_dim
    run_dict['grid_dim'] = 'default'

    ##block_dim
    run_dict['block_dim'] = 'default'

    ##device_id
    device_id = lines[28][lines[28].find(':') + 2:]
    run_dict['device_id'] = int(device_id)

    ##grid_multip
    grid_multip = lines[29][lines[29].find(':') + 2:]
    run_dict['grid_multip'] = int(grid_multip)

    ##decomposition
    decomposition = lines[30][lines[30].find(':') + 2:]
    run_dict['decomposition'] = int(decomposition)

    ##scaling_thresh
    scaling_thresh = lines[31][lines[31].find(':') + 2:]
    run_dict['scaling_thresh'] = int(scaling_thresh)

    ##host
    host = lines[32][lines[32].find(':') + 2:]
    run_dict['host'] = host

    ##synth_size ##synth_density ##synth_type
    synth_name = lines[20][lines[20].find(':') + 2:]

    synt_type = ''
    if(synth_name.find('real') != -1):
        synth_type = 'real'
    else:
        synth_type = 'int'
    
    synth_name = synth_name[synth_name.rfind('/') + 1:].replace(' ', '')

    size, density, _dum = synth_name.split('_')

    run_dict['synth_type'] = synth_type
    run_dict['synth_size'] = size
    run_dict['synth_density'] = density
    
    

    return run_dict



def extract_result_fields(run_dict, lines):

    
    res_lines = lines[34:]
    #print('res_lines', res_lines)

    results = []
    times = []

    first = True
    
    for item in res_lines:
        if(item.find('Result') != -1):
            _dum1, _dum2, _algo_name, _dum3, _res = item.split('|')
            run_dict['algo_name'] = _algo_name.strip(' ')
            result, time = _res.split("in")
            result = result.strip(' ')
            time = time.strip(' ')
            if not first:
                results.append(float(result))
                times.append(float(time))
            first = False

            

    run_dict['time'] = statistics.mean(times)
    run_dict['acc_result_avg'] = statistics.mean(results)
    run_dict['acc_max'] = max(results)
    run_dict['acc_min'] = min(results)
    run_dict['acc_stdev'] = statistics.stdev(results)
    
    return run_dict
    
    
def parse_one(one_run):

    reader = open(one_run)
    lines = reader.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].strip('\n')
    
    #for i in range (len(lines)):
        #print(i, lines[i])

    run_dict = dict.fromkeys(keys)
    
    run_dict = common_fields(run_dict, lines)
    run_dict = extract_result_fields(run_dict, lines)

    return run_dict

if __name__ == "__main__":
    #df = pd.DataFrame(columns = keys)
    
    
    files = get_files()
    #arun_dict = parse_one(files[0])
    
    #for key in arun_dict:
        #print('k', key, 'v', arun_dict[key])

    #print(arun_dict)

    dicts = []
    for i in range(len(files)):
        arun_dict = parse_one(files[i])
        dicts.append(arun_dict)
        #df.loc[i] = [*arun_dict.values()]
        #print([*arun_dict.values()])

    df = pd.DataFrame(data = dicts)



    #####SPARSE#####
    sparse_010 = df.loc[(df['sparse'] == 1) & (df['synth_density'] == '0.10') ]
    sparse_020 = df.loc[(df['sparse'] == 1) & (df['synth_density'] == '0.20') ]
    sparse_030 = df.loc[(df['sparse'] == 1) & (df['synth_density'] == '0.30') ]
    sparse_040 = df.loc[(df['sparse'] == 1) & (df['synth_density'] == '0.40') ]
    sparse_050 = df.loc[(df['sparse'] == 1) & (df['synth_density'] == '0.50') ]
    sparse_060 = df.loc[(df['sparse'] == 1) & (df['synth_density'] == '0.60') ]
    
    s10_pv = sparse_010.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    s20_pv = sparse_020.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    s30_pv = sparse_030.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    s40_pv = sparse_040.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    s50_pv = sparse_050.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    s60_pv = sparse_060.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')

    df_list = [s10_pv, s20_pv, s30_pv, s40_pv, s50_pv, s60_pv]

    nrow = 3
    ncol = 2
    fig, axes = plt.subplots(nrow,ncol, sharex=True, sharey=False)

    print(df_list)
    densities = ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60']
    
    count = 0
    for r in range(nrow):
        for c in range(ncol):
            df_list[count].plot(ax=axes[r,c], kind='bar', edgecolor='black', legend=False, width=0.9)
            axes[r,c].set_title('Density: ' + densities[count])
            count += 1

    lines = []
    labels = []
  
    
    Line, Label = axes[0,0].get_legend_handles_labels()
    print(Label)
    lines.extend(Line)
    labels.extend(Label)

    fig.subplots_adjust(left=0.067, bottom=0.134, right=0.987, top=0.917, wspace=0.127, hspace=0.2)
    fig.suptitle('Single GPU - Sparse')
    fig.text(0.01, 0.5, 'Time', ha='center', va='center', rotation='vertical')
    fig.legend(lines, labels, loc='lower center', ncol = 2)
        
    #####SPARSE#####


    #####DENSE#####
    
    dense_010 = df.loc[(df['dense'] == 1) & (df['synth_density'] == '0.10')]
    dense_020 = df.loc[(df['dense'] == 1) & (df['synth_density'] == '0.20')]
    dense_030 = df.loc[(df['dense'] == 1) & (df['synth_density'] == '0.30')]
    dense_040 = df.loc[(df['dense'] == 1) & (df['synth_density'] == '0.40')]
    dense_050 = df.loc[(df['dense'] == 1) & (df['synth_density'] == '0.50')]
    dense_060 = df.loc[(df['dense'] == 1) & (df['synth_density'] == '0.60')]
    
    d10_pv = dense_010.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    d20_pv = dense_020.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    d30_pv = dense_030.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    d40_pv = dense_040.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    d50_pv = dense_050.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    d60_pv = dense_060.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')

    df_list_2 = [d10_pv, d20_pv, d30_pv, d40_pv, d50_pv, d60_pv]

    nrow = 3
    ncol = 2
    fig2, axes2 = plt.subplots(nrow,ncol)

    print(df_list)

    count = 0
    for r in range(nrow):
        for c in range(ncol):
            df_list_2[count].plot(ax=axes2[r,c], kind='bar', edgecolor='black', legend=False, width=0.9)
            axes2[r,c].set_title('Density: ' + densities[count])
            count += 1


    lines = []
    labels = []
  
    
    Line, Label = axes2[0,0].get_legend_handles_labels()
    print(Label)
    lines.extend(Line)
    labels.extend(Label)

    fig2.subplots_adjust(left=0.067, bottom=0.134, right=0.987, top=0.917, wspace=0.127, hspace=0.2)
    fig2.suptitle('Single GPU - Dense')
    fig2.text(0.01, 0.5, 'Time', ha='center', va='center', rotation='vertical')
    fig2.legend(lines, labels, loc='lower center', ncol = 2)

    plt.tight_layout()
    plt.show()
    
    #####DENSE#####


    print(df.pattern.unique())
