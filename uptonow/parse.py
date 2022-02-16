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
        'mtype',
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



paired_algo_colors = {'gpu_perman_xglobal': '#B4BCD6', ##karpfenlau1
                      'gpu_perman_xlocal': '#8290BB',
                      'gpu_perman_xshared': '#586BA4',
                      'gpu_perman_xshared_coalescing': '#3E5496',
                      'gpu_perman_xshared_coalescing_mshared': '#324376',
                      
                      'gpu_perman_xlocal_sparse': '#BC7A8F', #bordeaux1
                      'gpu_perman_xshared_sparse': '#A54D69',
                      'gpu_perman_xshared_coalescing_sparse': '#8E2043',
                      'gpu_perman_xshared_coalescing_mshared_sparse': '#771434',
                      'gpu_perman_xshared_coalescing_mshared_skipper': '#EFDC60' #signal2
}
    



algo_order_sequence = ['gpu_perman_xlocal' ,
                       'gpu_perman_xshared',
                       'gpu_perman_xshared_coalescing',
                       'gpu_perman_xshared_coalescing_mshared',
                       'gpu_perman_xlocal_sparse' ,
                       'gpu_perman_xshared_sparse',
                       'gpu_perman_xshared_coalescing_sparse',
                       'gpu_perman_xshared_coalescing_mshared_sparse',
                       'gpu_perman_xshared_coalescing_mshared_skipper']

algo_order_pair = ['gpu_perman_xlocal' ,
                   'gpu_perman_xlocal_sparse' ,
                   'gpu_perman_xshared',
                   'gpu_perman_xshared_sparse',
                   'gpu_perman_xshared_coalescing',
                   'gpu_perman_xshared_coalescing_sparse',
                   'gpu_perman_xshared_coalescing_mshared',
                   'gpu_perman_xshared_coalescing_mshared_sparse',
                   'gpu_perman_xshared_coalescing_mshared_skipper']


              

def column_order(columns, chosen_order):

    print('columns:', columns)
    print('chosen_order:', chosen_order)
    order = []
    
    for item in chosen_order:
        if item in columns:
            order.append(item)
            
    print('returning:', order)
    return order

def color_order(columns):

    colors = []

    for item in columns:        
        colors.append(paired_algo_colors[item])
        

    return colors


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
    run_dict['mtype'] = ttype

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

def two_charts(df, include, exclude):

    
    include_str = ''
    exclude_str = ''

    #for item in include:
        #print('include:', item)

    #for item in include:
        #print('exclude:', item)
    
    for i in range(len(include)):
        #print('include',include[i][0], include[i][1])
        df = df.loc[(df[include[i][0]] == include[i][1])]
        #print('include:', df)
        include_str += include[i][0] + '-' + str(include[i][1]) + ' '

    for i in range(len(exclude)):
        df = df.loc[(df[exclude[i][0]] != exclude[i][1])]
        exclude_str += exclude[i][0] + '-' + str(exclude[i][1]) + ' '

        
        

    #####SPARSE#####
    sparse_010 = df.loc[(df['sparse'] == 1) & (df['synth_density'] == '0.10')]
    sparse_020 = df.loc[(df['sparse'] == 1) & (df['synth_density'] == '0.20')]
    sparse_030 = df.loc[(df['sparse'] == 1) & (df['synth_density'] == '0.30')]
    sparse_040 = df.loc[(df['sparse'] == 1) & (df['synth_density'] == '0.40')]
    sparse_050 = df.loc[(df['sparse'] == 1) & (df['synth_density'] == '0.50')]
    sparse_060 = df.loc[(df['sparse'] == 1) & (df['synth_density'] == '0.60')]
    
    s10_pv = sparse_010.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    s20_pv = sparse_020.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    s30_pv = sparse_030.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    s40_pv = sparse_040.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    s50_pv = sparse_050.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    s60_pv = sparse_060.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')

    df_list = [s10_pv, s20_pv, s30_pv, s40_pv, s50_pv, s60_pv]
    color_list = []
    
    for i in range(len(df_list)):
        df_list[i] = df_list[i].reindex(columns = column_order(df_list[i].columns.values, algo_order_pair))
        color_list.append(color_order(df_list[i].columns.values)) ##Sends an array, asks for an array

    nrow = 3
    ncol = 2
    fig, axes = plt.subplots(nrow,ncol, sharex=True, sharey=False)

    #print(df_list)
    densities = ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60']
    
    count = 0
    for r in range(nrow):
        for c in range(ncol):
            columns_and_colors = zip(df_list[i].columns.values, color_list[i])
            df_list[count].plot.bar(ax=axes[r,c], edgecolor='black', legend=False, width=0.9, color=[cc[1] for cc in columns_and_colors])
            axes[r,c].set_title('Density: ' + densities[count])
            count += 1

    hatch = ['*', '.', '--', '+']
    bars = axes[0,0].patches
    for i in range(len(bars)):
        bars[i].set_hatch(hatch[i%4])

    lines = []
    labels = []
  
    
    Line, Label = axes[0,0].get_legend_handles_labels()
    #print(Label)
    lines.extend(Line)
    labels.extend(Label)

    fig.subplots_adjust(left=0.067, bottom=0.134, right=0.987, top=0.917, wspace=0.127, hspace=0.2)
    fig.suptitle('Single GPU - Sparse' + ' +: ' + include_str + ' -:' + exclude_str)
    fig.text(0.01, 0.5, 'Time', ha='center', va='center', rotation='vertical')
    fig.legend(lines, labels, loc='lower center', ncol = 2)
        
    #####SPARSE#####


    #####DENSE#####

    df_no_glob = df.loc[(df['algo_name'] != 'gpu_perman_xglobal')]
    
    dense_010 = df_no_glob.loc[(df['dense'] == 1) & (df['synth_density'] == '0.10')]
    dense_020 = df_no_glob.loc[(df['dense'] == 1) & (df['synth_density'] == '0.20')]
    dense_030 = df_no_glob.loc[(df['dense'] == 1) & (df['synth_density'] == '0.30')]
    dense_040 = df_no_glob.loc[(df['dense'] == 1) & (df['synth_density'] == '0.40')]
    dense_050 = df_no_glob.loc[(df['dense'] == 1) & (df['synth_density'] == '0.50')]
    dense_060 = df_no_glob.loc[(df['dense'] == 1) & (df['synth_density'] == '0.60')]
    
    d10_pv = dense_010.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    d20_pv = dense_020.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    d30_pv = dense_030.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    d40_pv = dense_040.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    d50_pv = dense_050.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    d60_pv = dense_060.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')

    df_list_2 = [d10_pv, d20_pv, d30_pv, d40_pv, d50_pv, d60_pv]
    color_list_2 = []
    
    for i in range(len(df_list_2)):
        df_list_2[i] = df_list_2[i].reindex(columns = column_order(df_list_2[i].columns.values, algo_order_pair))
        color_list_2.append(color_order(df_list[i].columns.values)) ##Sends an array, asks for an array

    nrow = 3
    ncol = 2
    fig2, axes2 = plt.subplots(nrow, ncol, sharex=True, sharey=False)

    #print(df_list)

    count = 0
    for r in range(nrow):
        for c in range(ncol):
            columns_and_colors = zip(df_list_2[i].columns.values, color_list_2[i])
            df_list_2[count].plot(ax=axes2[r,c], kind='bar', edgecolor='black', legend=False, width=0.9, color=[cc[1] for cc in columns_and_colors])
            axes2[r,c].set_title('Density: ' + densities[count])
            count += 1


    lines = []
    labels = []
  
    
    Line, Label = axes2[0,0].get_legend_handles_labels()
    #print(Label)
    lines.extend(Line)
    labels.extend(Label)

    fig2.subplots_adjust(left=0.067, bottom=0.134, right=0.987, top=0.917, wspace=0.127, hspace=0.2)
    fig2.suptitle('Single GPU - Dense' + ' +: ' + include_str + ' -: ' + exclude_str)
    fig2.text(0.01, 0.5, 'Time', ha='center', va='center', rotation='vertical')
    fig2.legend(lines, labels, loc='lower center', ncol = 2)

    
def one_chart(df, include, exclude):

    df = df.loc[(df['algo_name'] != 'gpu_perman_xglobal')]
    
    include_str = ''
    exclude_str = ''

    #for item in include:
        #print('include:', item)

    #for item in include:
        #print('exclude:', item)


    
    for i in range(len(include)):
        #print('include',include[i][0], include[i][1])
        df = df.loc[(df[include[i][0]] == include[i][1])]
        #print('include:', df)
        include_str += include[i][0] + '-' + str(include[i][1]) + ' '

    for i in range(len(exclude)):
        df = df.loc[(df[exclude[i][0]] != exclude[i][1])]
        exclude_str += exclude[i][0] + '-' + str(exclude[i][1]) + ' '

    #####SPARSE#####
    sparse_010 = df.loc[(df['synth_density'] == '0.10')]
    sparse_020 = df.loc[(df['synth_density'] == '0.20')]
    sparse_030 = df.loc[(df['synth_density'] == '0.30')]
    sparse_040 = df.loc[(df['synth_density'] == '0.40')]
    sparse_050 = df.loc[(df['synth_density'] == '0.50')]
    sparse_060 = df.loc[(df['synth_density'] == '0.60')]
    
    s10_pv = sparse_010.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    s20_pv = sparse_020.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    s30_pv = sparse_030.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    s40_pv = sparse_040.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    s50_pv = sparse_050.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')
    s60_pv = sparse_060.pivot_table(index=['synth_size'], columns= 'algo_name', values='time')

    df_list = [s10_pv, s20_pv, s30_pv, s40_pv, s50_pv, s60_pv]
    color_list = []

    ##Arranging order and look
    for i in range(len(df_list)):
        df_list[i] = df_list[i].reindex(columns = column_order(df_list[i].columns.values, algo_order_pair))
        color_list.append(color_order(df_list[i].columns.values)) ##Sends an array, asks for an array

    print(color_list)
        
    ##Arranging order and look
        
    nrow = 3
    ncol = 2
    fig, axes = plt.subplots(nrow,ncol, sharex=True, sharey=False)

    #print(df_list)
    densities = ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60']
    
    count = 0
    for r in range(nrow):
        for c in range(ncol):
            columns_and_colors = zip(df_list[i].columns.values, color_list[i])
            df_list[count].plot(ax=axes[r,c], kind='bar', edgecolor='black', legend=False, width=0.9, color=[cc[1] for cc in columns_and_colors])
            axes[r,c].set_title('Density: ' + densities[count])
            count += 1

    lines = []
    labels = []
  
    
    Line, Label = axes[0,0].get_legend_handles_labels()
    print(Label)
    lines.extend(Line)
    labels.extend(Label)

    fig.subplots_adjust(left=0.067, bottom=0.134, right=0.987, top=0.917, wspace=0.127, hspace=0.2)
    fig.suptitle('Single GPU - Sparse vs Dense' + ' +: ' + include_str + ' -: ' + exclude_str)
    fig.text(0.01, 0.5, 'Time', ha='center', va='center', rotation='vertical')
    fig.legend(lines, labels, loc='lower center', ncol = 4)
        
    #####SPARSE#####

    
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
            try:
                result, time = _res.split(" in ")
            except:
                print(_res)
                exit(1)
                
            result = result.strip(' ')
            time = time.strip(' ')
            if not first:
                results.append(float(result))
                times.append(float(time))
            first = False

            

    try:
        run_dict['time'] = statistics.mean(times)
        run_dict['acc_result_avg'] = statistics.mean(results)
        run_dict['acc_max'] = max(results)
        run_dict['acc_min'] = min(results)
        run_dict['acc_stdev'] = statistics.stdev(results)

    except:
        print(lines[0])
        #for item in lines:
        #print(item)
        #exit(1)
    
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
    algos = ['gpu_perman_xglobal',
             'gpu_perman_xlocal'
             'gpu_perman_xshared',
             'gpu_perman_xshared_coalescing',
             'gpu_perman_xshared_coalescing_mshared'
             'gpu_perman_xlocal',
             'gpu_perman_xlocal_sparse',
             'gpu_perman_xshared_sparse',
             'gpu_perman_xshared_coalescing_sparse',
             'gpu_perman_xshared_coalescing_mshared',
             'gpu_perman_xshared_coalescing_mshared_sparse',
             'gpu_perman_xshared_coalescing_mshared_skipper']
    
    #mapping = {algos: i for i, algos in enumerate(algos)}
    #df['algo_name'] = df['algo_name'].map(mapping)
    #df.to_excel("try0.xlsx")
    #df = df.loc[(df['mtype'] == 'real')]
    #print('MAIN:', df)
    #exit(1)
    
    #####DENSE#####

    include = []
    exclude = []
    
    #two_charts(df, [],[])
    #two_charts(df, [['pattern', 0], ['synth_type', 'real']], [['halfstore', '1'], ['halfprec', '1']])
    #two_charts(df, [['pattern', 0], ['synth_type', 'int'], ['ordering', 0]], [['halfstore', '1'], ['halfprec', '1']])
    #two_charts(df, [['pattern', 0], ['synth_type', 'int'], ['ordering', 1]], [['halfstore', '1'], ['halfprec', '1']])
    #two_charts(df, [['pattern', 0], ['synth_type', 'real'], ['ordering', 2]], [['halfstore', '1'], ['halfprec', '1']])
    #two_charts(df, [['pattern', 1], ['synth_type', 'real']], [['halfstore', '1'], ['halfprec', '1']])
    #two_charts(df, [['pattern', 1], ['synth_type', 'int']], [['halfstore', '1'], ['halfprec', '1']])


    two_charts(df, [['pattern', 0], ['synth_type', 'real']], [['halfstore', '1'], ['halfprec', '1'], ['algo_name', 'gpu_perman_xshared_coalescing_mshared_skipper']])
    two_charts(df, [['pattern', 0], ['synth_type', 'real'], ['ordering', 1]], [['halfstore', '1'], ['halfprec', '1']])
    
    one_chart(df, [['pattern', 0], ['synth_type', 'real']], [['halfstore', '1'], ['halfprec', '1'], ['algo_name', 'gpu_perman_xshared_coalescing_mshared_skipper']])
    one_chart(df, [['pattern', 0], ['synth_type', 'real'], ['ordering', 1]], [['halfstore', '1'], ['halfprec', '1']])
    #one_chart(df, [['pattern', 0], ['synth_type', 'int']], [['halfstore', '1'], ['halfprec', '1']])
    #one_chart(df, [['pattern', 1], ['synth_type', 'real']], [['halfstore', '1'], ['halfprec', '1']])
    ##one_chart(df, [['pattern', 1], ['synth_type', 'int']], [['halfstore', '1'], ['halfprec', '1']])See, no data, not working
    

    

    #plt.tight_layout()
    plt.show()

    print('type: ', df.mtype.unique())
    print('patterns: ', df.pattern.unique())
    print('realw_name: ', df.realw_name.unique())
    #print('algo_name:', df.algo_name.unique())
    print('ordering:', df.ordering.unique())
    print('')
    #df.reindex(columns=column_order(df.algo_name.unique()))
    #print('ordered: ', column_order(df.algo_name.unique()))
    

    #print(df.shape())
    #df = df.loc[(df['ordering'] == 2)]
    #print(df.head())
    


    
