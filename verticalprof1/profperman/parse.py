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
        'perman_name',
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



paired_algo_colors = {'regular_perman': '#73787E', ##grau4
                      'plain_perman': '#71D1CC',
                      'plaintex_perman': '#067E79',
                      'transtride_perman': '#771434'} ##bordeaux4

    

version_order_sequence = ['regular_perman',
                          'plain_perman',
                          'plaintex_perman',
                          'transtride_perman']

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
    
    ##perman_name
    perman_name = lines[0].split(' ')[1].strip('./').replace(' ', '')
    if(perman_name == 'gpu_perman'):
        perman_name = 'regular_perman'
    run_dict['perman_name'] = perman_name
    
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

def recursive_select_help(df, in_st, include, depth, max_depth):

    ret = df.loc[(df[include[depth][0]] == include[depth][1])]
    in_st += include[depth][0] + '-' + str(include[depth][1]) + ' '

    print('include:', include[depth])
    print('depth:', depth, 'shape:', ret.shape)
    
    if(depth == max_depth):
        return ret, in_st
    else:
        return recursive_select_help(ret, in_st, include, depth+1, max_depth)
    
    
def recursive_select(df, bool_include):

    include_str = ''

    if(len(bool_include) == 0):
        return df, include_str
    
    return recursive_select_help(df, include_str, bool_include, 0, len(bool_include)-1)

def prepare_dataset(df, bool_include, slice_include, exclude):

    #include_str = ''
    exclude_str = ''
    
    exclude_copy_df = df.copy(deep=True)

    
    for i in range(len(exclude)):
        exclude_copy_df = exclude_copy_df.loc[(exclude_copy_df[exclude[i][0]] != exclude[i][1])]
        exclude_str += exclude[i][0] + '-' + str(exclude[i][1]) + ' '

    #print('Exclude unique:', exclude_copy_df['algo_name'].unique())
    print('Exclude: ', exclude)
    print('Exclude unique:', exclude_copy_df['perman_name'].unique())

    include_copy_df = exclude_copy_df.copy(deep=True)
    remaining_df = pd.DataFrame(data = [], columns = keys)

    
    remaining_df, include_str = recursive_select(include_copy_df, bool_include)
    print('remaining_df: ', remaining_df.shape)
    slice_include_df = pd.DataFrame(data = [], columns = keys)
    
    for i in range(len(slice_include)):
        iter_df = remaining_df.loc[(remaining_df[slice_include[i][0]] == slice_include[i][1])]
        slice_include_df = pd.concat([slice_include_df, iter_df], ignore_index=True)
        #include_str += slice_include[i][0] + '-' + str(slice_include[i][1]) + ' '
    

    print('###############################################################')
    df = slice_include_df
    print('bool_include_df shape:', df.shape)
    print('type: ', df.mtype.unique())
    print('patterns: ', df.pattern.unique())
    print('realw_name: ', df.realw_name.unique())
    print('algo_name:', df.algo_name.unique())
    print('ordering:', df.ordering.unique())
    print('densities:', df.synth_density.unique())
    print('sparse:', df['sparse'].unique())
    print('dense:', df['dense'].unique())
    print('sizes:', df.synth_size.unique())
    print(df.head(50))
    print('###############################################################')

    return df, include_str, exclude_str


def general_graph(df, x, y, bars, multidimname, multidimval, bool_include, slice_include, exclude, title):
    ##x -> size
    ##y -> time (accuracy ?)
    ##multidim -> density

    df, include_str, exclude_str = prepare_dataset(df, bool_include, slice_include, exclude)
    
    df_list = []
    for val in multidimval:
        df_list.append(df.loc[df[multidimname] == val])
                

    pivots = []
    for df in df_list:
        pivots.append(df.pivot_table(index=[x], columns=bars, values=y))


    color_list = []
    for i in range(len(pivots)):
        pivots[i] = pivots[i].reindex(columns = column_order(pivots[i].columns.values, version_order_sequence))
        print('pivots', i, 'columns')
        color_list.append(color_order(pivots[i].columns.values)) ##Sends an array, asks for an array


    nrow = int(len(multidimval) / 2)
    ncol = 2
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=False)

    for r in range(nrow):
        for c in range(ncol):
            columns_and_colors = zip(pivots[i].columns.values, color_list[i])
            pivots[r*ncol+c].plot.bar(ax=axes[r,c], edgecolor='black', width=0.9,
                                      color=[cc[1] for cc in columns_and_colors],
                                      legend=False)
            axes[r,c].set_title('Density: ' + multidimval[r*ncol+c])
            ax = axes[r,c]
            #for container in ax.containers:
                #ax.bar_label(container)
            #plt.show()

    lines = []
    labels = []

    Line, Label = axes[0,0].get_legend_handles_labels()
    lines.extend(Line)
    labels.extend(Label)

    fig.subplots_adjust(left=0.067, bottom=0.134, right=0.987, top=0.917, wspace=0.127, hspace=0.2)
    #fig.suptitle(title + ' +: ' + include_str + ' -:' + exclude_str)
    fig.suptitle(title)
    fig.text(0.01, 0.5, 'Time', ha='center', va='center', rotation='vertical')
    fig.legend(lines, labels, loc='lower center', ncol = 2)

    

    
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
        
    files = get_files()
    
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
    

    print(df.head())

    version_include = [['perman_name', 'regular_perman'],
                       ['perman_name', 'plain_perman'],
                       ['perman_name', 'plaintex_perman'],
                       ['perman_name', 'transtride_perman']]

    
    small_exclude = [['synth_size', '38'],
                     ['synth_size', '37'],
                     ['synth_size', '36'],
                     ['synth_size', '35'],
                     ['synth_size', '34'],
                     ['synth_size', '33']]

    mid_exclude = [['synth_size', '30'],
                     ['synth_size', '31'],
                     ['synth_size', '32'],
                     ['synth_size', '36'],
                     ['synth_size', '37'],
                     ['synth_size', '38']]

    big_exclude = [['synth_size', '30'],
                     ['synth_size', '31'],
                     ['synth_size', '32'],
                     ['synth_size', '33'],
                     ['synth_size', '34'],
                     ['synth_size', '35']]
    
    bool_include = []
    
    
    densities = ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80']
    
    general_graph(df, 'synth_size', 'time', 'perman_name', 'synth_density', densities,
    bool_include, version_include, small_exclude, 'Memory Access Patterns - Dense - Smaller')

    general_graph(df, 'synth_size', 'time', 'perman_name', 'synth_density', densities,
    bool_include, version_include, mid_exclude, 'Memory Access Patterns - Dense - Medium')

    general_graph(df, 'synth_size', 'time', 'perman_name', 'synth_density', densities,
    bool_include, version_include, big_exclude, 'Memory Access Patterns - Dense - Bigger')

    general_graph(df, 'synth_size', 'time', 'perman_name', 'synth_density', densities,
    bool_include, version_include, small_exclude, 'THIS IS EMPTY')

    
    
    plt.tight_layout()
    plt.show()

    print('type: ', df.mtype.unique())
    print('patterns: ', df.pattern.unique())
    print('realw_name: ', df.realw_name.unique())
    print('algo_name:', df.algo_name.unique())
    print('ordering:', df.ordering.unique())
    print('densities:', df.synth_density.unique())
    print('decomposition:', df.decomposition.unique())
    print('synth_size:', df.synth_size.unique())
    print('perman_name:', df.perman_name.unique())
    


    
