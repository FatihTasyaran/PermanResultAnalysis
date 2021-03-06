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



paired_algo_colors = {'gpu_perman_xglobal': '#B4BCD6', ##karpfenblau1
                      'gpu_perman_xlocal': '#8290BB',
                      'gpu_perman_xshared': '#586BA4',
                      'gpu_perman_xshared_coalescing': '#3E5496',
                      'gpu_perman_xshared_coalescing_mshared': '#324376',
                      
                      'gpu_perman_xlocal_sparse': '#BC7A8F', #bordeaux1
                      'gpu_perman_xshared_sparse': '#A54D69',
                      'gpu_perman_xshared_coalescing_sparse': '#8E2043',
                      'gpu_perman_xshared_coalescing_mshared_sparse': '#771434',
                      'gpu_perman_xshared_coalescing_mshared_skipper': '#EFDC60', #signal2
                      
                      'gpu_perman_xlocal_sparse SparseOrder': '#BC7A8F', #bordeaux1
                      'gpu_perman_xshared_sparse SparseOrder': '#A54D69',
                      'gpu_perman_xshared_coalescing_sparse SparseOrder': '#8E2043',
                      'gpu_perman_xshared_coalescing_mshared_sparse SparseOrder': '#771434',
                      'gpu_perman_xshared_coalescing_mshared_skipper SkipOrder': '#EFDC60', #signal2

                      ##CPU algo colors
                      'parallel_perman': '#71D1CC', ##seegruen1
                      'parallel_perman_sparse': '#0A9086', ##seegruen4
                      'parallel_perman_sparse SparseOrder': '#0A9086', ##seegruen4
                      'parallel_skip_perman_balanced': '#FEA090', ##peach4
                      'parallel_skip_perman_balanced SkipOrder': '#FEA090' ##peach4
}

paired_algo_hatches = {'gpu_perman_xglobal': '',
                       'gpu_perman_xlocal': '',
                       'gpu_perman_xshared': '',
                       'gpu_perman_xshared_coalescing': '',
                       'gpu_perman_xshared_coalescing_mshared': '',
                      
                       'gpu_perman_xlocal_sparse': '',
                       'gpu_perman_xshared_sparse': '',
                       'gpu_perman_xshared_coalescing_sparse': '',
                       'gpu_perman_xshared_coalescing_mshared_sparse': '',
                       'gpu_perman_xshared_coalescing_mshared_skipper': '',
                       
                       'gpu_perman_xlocal_sparse SparseOrder': '///',
                       'gpu_perman_xshared_sparse SparseOrder': '///',
                       'gpu_perman_xshared_coalescing_sparse SparseOrder': '///',
                       'gpu_perman_xshared_coalescing_mshared_sparse SparseOrder': '///',
                       'gpu_perman_xshared_coalescing_mshared_skipper SkipOrder': '///',

                       'parallel_perman' : '',

                       'parallel_perman_sparse': '',
                       'parallel_skip_perman_balanced': '',
                       
                       'parallel_perman_sparse SparseOrder': '///',
                       'parallel_skip_perman_balanced SkipOrder': '///'
}
    



algo_order_sequence = ['parallel_perman',
                       'parallel_perman_sparse',
                       'parallel_skip_perman_balanced',
                       'gpu_perman_xlocal' ,
                       'gpu_perman_xshared',
                       'gpu_perman_xshared_coalescing',
                       'gpu_perman_xshared_coalescing_mshared',
                       'gpu_perman_xlocal_sparse' ,
                       'gpu_perman_xlocal_sparse SparseOrder' ,
                       'gpu_perman_xshared_sparse',
                       'gpu_perman_xshared_sparse SparseOrder',
                       'gpu_perman_xshared_coalescing_sparse',
                       'gpu_perman_xshared_coalescing_sparse SparseOrder',
                       'gpu_perman_xshared_coalescing_mshared_sparse',
                       'gpu_perman_xshared_coalescing_mshared_sparse SparseOrder',
                       'gpu_perman_xshared_coalescing_mshared_skipper',
                       'gpu_perman_xshared_coalescing_mshared_skipper SkipOrder']



algo_order_pair = ['parallel_perman',
                   'parallel_perman_sparse',
                   'parallel_perman_sparse SparseOrder',
                   'parallel_skip_perman_balanced',
                   'parallel_skip_perman_balanced SkipOrder',
                   'gpu_perman_xlocal' ,
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

def hatch_order(columns):

    hatches = []

    for item in columns:
        hatches.append(paired_algo_hatches[item])


    return hatches


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
    
    return recursive_select_help(df, include_str, bool_include, 0, len(bool_include)-1)

def prepare_dataset(df, bool_include, slice_include, exclude):

    #include_str = ''
    exclude_str = ''

    exclude_copy_df = df.copy(deep=True)

    
    for i in range(len(exclude)):
        exclude_copy_df = exclude_copy_df.loc[(exclude_copy_df[exclude[i][0]] != exclude[i][1])]
        exclude_str += exclude[i][0] + '-' + str(exclude[i][1]) + ' '

    print('Exclude unique:', exclude_copy_df['algo_name'].unique())

    include_copy_df = exclude_copy_df.copy(deep=True)
    remaining_df = pd.DataFrame(data = [], columns = keys)
        
    #for i in range(len(bool_include)):
        #iter_df = include_copy_df.loc[(include_copy_df[bool_include[i][0]] == bool_include[i][1])]
        #print('iterdf: ', iter_df.shape)
        #print('bool include: ', bool_include[i][0])
        #print('unique: ', iter_df[bool_include[i][0]].unique(), iter_df[bool_include[3][0]].unique())
        #remaining_df = pd.concat([remaining_df, iter_df], ignore_index=True)
        #include_str += bool_include[i][0] + '-' + str(bool_include[i][1]) + ' '
        
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

def two_charts(df, bool_include, slice_include, exclude):

    
    df, include_str, exclude_str = prepare_dataset(df, bool_include, slice_include, exclude)
        

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
            #ax = axes[r,c]
            #for container in ax.containers:
                #ax.bar_label(container)

    ###HATCHES HERE
    #hatch = ['*', '.', '--', '+']
    #bars = axes[0,0].patches
    #for i in range(len(bars)):
        #bars[i].set_hatch(hatch[i%4])

    

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
        color_list_2.append(color_order(df_list_2[i].columns.values)) ##Sends an array, asks for an array

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

    
def one_chart(df, bool_include, slice_include, exclude):

    df = df.loc[(df['algo_name'] != 'gpu_perman_xglobal')]
    
    df, include_str, exclude_str = prepare_dataset(df, bool_include, slice_include, exclude)

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
    fig.legend(lines, labels, loc='lower center', ncol = 5)
        
    #####SPARSE#####

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
        pivots[i] = pivots[i].reindex(columns = column_order(pivots[i].columns.values, algo_order_pair))
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


def ugly_cpu_bool(num):

    yes = [8,9,10,11, 16,17,18,19]

    if num in yes:
        return True
    else:
        return False
    

    

def cpus_graph(df, x, y, bars, multidimname, multidimval, bool_include, slice_include, exclude, title):
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
        pivots[i] = pivots[i].reindex(columns = column_order(pivots[i].columns.values, algo_order_pair))
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

    for r in range(nrow):
        for c in range(ncol):
            bars = axes[r,c].patches
            for i in range(len(bars)):
                if(ugly_cpu_bool(i)):
                    bars[i].set_hatch('///')

    lines = []
    labels = []

    Line, Label = axes[0,0].get_legend_handles_labels()
    lines.extend(Line)
    labels.extend(Label)

    labels = ['ParRyser',
              'SpaRyser',
              'SpaRyser + SparseOrder',
              'Skipper',
              'Skipper + SkipOrder']


    fig.subplots_adjust(left=0.067, bottom=0.134, right=0.987, top=0.917, wspace=0.127, hspace=0.2)
    #fig.suptitle(title + ' +: ' + include_str + ' -:' + exclude_str)
    fig.suptitle(title)
    fig.text(0.01, 0.5, 'Time', ha='center', va='center', rotation='vertical')
    fig.legend(lines, labels, loc='lower center', ncol = 5)


    

    

#Indeed
def ugly_bool(num):

    yes = [4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31]
    no = [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]

    if num in yes:
        return True
    elif num in no:
        return False
    else:
        print('num:', num)
        #exit(1)
        
    


def sparse_order_exclusive(df, x, y, bars, multidimname, multidimval, bool_include, slice_include, exclude, title):
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
    hatch_list = []
    for i in range(len(pivots)):
        pivots[i] = pivots[i].reindex(columns = column_order(pivots[i].columns.values, algo_order_sequence))
        print('pivots', i, 'columns')
        color_list.append(color_order(pivots[i].columns.values)) ##Sends an array, asks for an array
        hatch_list.append(hatch_order(pivots[i].columns.values)) ##Sends an array, asks for an array


    print('HATCH LIST:', hatch_list)
        
    nrow = int(len(multidimval) / 2)
    ncol = 2
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=False)

    for r in range(nrow):
        for c in range(ncol):
            columns_and_colors = zip(pivots[r*ncol+c].columns.values, color_list[r*ncol+c])
            columns_and_hatches = zip(pivots[r*ncol+c].columns.values, hatch_list[r*ncol+c])
            pivots[r*ncol+c].plot.bar(ax=axes[r,c], edgecolor='black', width=0.9,
                                      color=[cc[1] for cc in columns_and_colors],
                                      legend=False)
            axes[r,c].set_title('Density: ' + multidimval[r*ncol+c])
            ax = axes[r,c]
            #for container in ax.containers:
                #ax.bar_label(container)
            #plt.show()



    for r in range(nrow):
        for c in range(ncol):
            bars = axes[r,c].patches
            for i in range(len(bars)):
                if(ugly_bool(i)):
                    bars[i].set_hatch('///')

    lines = []
    labels = []

    Line, Label = axes[0,0].get_legend_handles_labels()
    lines.extend(Line)
    labels.extend(Label)

    print('lines:', lines)
    print('labels:', labels)
    labels = ['gpu_perman_xshared_sparse',
              'gpu_perman_xshared_coalescing_sparse',
              'gpu_perman_xshared_coalescing_mshared_sparse',
              'gpu_perman_xshared_sparse SparseOrder',
              'gpu_perman_xshared_coalescing_sparse SparseOrder',
              'gpu_perman_xshared_coalescing_mshared_sparse SparseOrder']

    one = lines[1]
    two = lines[2]
    three = lines[3]
    four = lines[4]

    lines[3] = one
    lines[1] = two
    lines[2] = four
    lines[4] = three
    
    #lines[1] = lines[2]
    #lines[5] = lines[1]
    #exit(1)

    fig.subplots_adjust(left=0.067, bottom=0.134, right=0.987, top=0.917, wspace=0.127, hspace=0.2)
    #fig.suptitle(title + ' +: ' + include_str + ' -:' + exclude_str)
    fig.suptitle(title)
    fig.text(0.01, 0.5, 'Time', ha='center', va='center', rotation='vertical')
    fig.legend(lines, labels, loc='lower center', ncol = 2)

    
def sparses_exclusive(df, x, y, bars, multidimname, multidimval, bool_include, slice_include, exclude, title):
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
    hatch_list = []
    for i in range(len(pivots)):
        pivots[i] = pivots[i].reindex(columns = column_order(pivots[i].columns.values, algo_order_sequence))
        print('pivots', i, 'columns')
        color_list.append(color_order(pivots[i].columns.values)) ##Sends an array, asks for an array
        hatch_list.append(hatch_order(pivots[i].columns.values)) ##Sends an array, asks for an array


    print('HATCH LIST:', hatch_list)
        
    nrow = int(len(multidimval) / 2)
    ncol = 2
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=False)

    for r in range(nrow):
        for c in range(ncol):
            columns_and_colors = zip(pivots[r*ncol+c].columns.values, color_list[r*ncol+c])
            columns_and_hatches = zip(pivots[r*ncol+c].columns.values, hatch_list[r*ncol+c])
            pivots[r*ncol+c].plot.bar(ax=axes[r,c], edgecolor='black', width=0.9,
                                      color=[cc[1] for cc in columns_and_colors],
                                      legend=False)
            axes[r,c].set_title('Density: ' + multidimval[r*ncol+c])
            ax = axes[r,c]
            #for container in ax.containers:
                #ax.bar_label(container)
            #plt.show()



    for r in range(nrow):
        for c in range(ncol):
            bars = axes[r,c].patches
            for i in range(len(bars)):
                if(ugly_bool(i)):
                    bars[i].set_hatch('///')

    lines = []
    labels = []

    Line, Label = axes[0,0].get_legend_handles_labels()
    lines.extend(Line)
    labels.extend(Label)

    print('lines:', lines)
    print('labels:', labels)
    #labels = ['gpu_perman_xshared_sparse',
    #'gpu_perman_xshared_coalescing_sparse',
    #'gpu_perman_xshared_coalescing_mshared_sparse',
    #'gpu_perman_xshared_sparse SparseOrder',
    #'gpu_perman_xshared_coalescing_sparse SparseOrder',
    #'gpu_perman_xshared_coalescing_mshared_sparse SparseOrder']

    #one = lines[1]
    #two = lines[2]
    #three = lines[3]
    #four = lines[4]

    #lines[3] = one
    #lines[1] = two
    #lines[2] = four
    #lines[4] = three
    
    #lines[1] = lines[2]
    #lines[5] = lines[1]
    #exit(1)

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
        print('Results missing:', lines[0])
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
    
        
    #####DENSE#####
    
    #general_graph(df, 'synth_size', 'time', 'algo_name', 'synth_density', densities,
    #all_include, slice_include_smaller, exclude, 'Single GPU - Sparse vs Dense- Smaller')
    

    
    

    densities = ['0.10', '0.20', '0.30', '0.40', '0.50']
    include = [['decomposition', 0], ['pattern', 0], ['quadprec', 0], ['quadstore', 0], ['scaling_thresh', -1], ['mtype', 'double']]
    pattern_include = [['decomposition', 0], ['pattern', 1], ['quadprec', 0], ['quadstore', 0], ['scaling_thresh', -1]]
    
    slice_include = [#['synth_size', '30'],
                     #['synth_size', '31'],
                     ['synth_size', '32'],
                     ['synth_size', '33'],
                     ['synth_size', '34'],
                     ['synth_size', '35']]
                     #['synth_size', '36']]
                     
    exclude = []


    
    parallel_include = [['decomposition', 0], ['pattern', 0], ['quadprec', 0], ['quadstore', 0], ['scaling_thresh', -1], ['algo_name', 'parallel_perman'], ['mtype', 'double']]
    parallel_pattern_include = [['algo_name', 'parallel_perman'], ['pattern', 1], ['decomposition', 0], ['pattern', 0], ['quadprec', 0], ['quadstore', 0], ['scali\
ng_thresh', -1]]

    sparse_include = [['decomposition', 0], ['pattern', 0], ['quadprec', 0], ['quadstore', 0], ['scaling_thresh', -1], ['mtype', 'double'], ['ordering', 0]]
    sparse_pattern_include = [['decomposition', 0], ['pattern', 1], ['quadprec', 0], ['quadstore', 0], ['scaling_thresh', -1], ['ordering', 0]]
    
    sparse_order_include = [['decomposition', 0], ['pattern', 0], ['quadprec', 0], ['quadstore', 0], ['scaling_thresh', -1], ['mtype', 'double'], ['ordering', 1]]
    sparse_order_pattern_include = [['decomposition', 0], ['pattern', 1], ['quadprec', 0], ['quadstore', 0], ['scaling_thresh', -1], ['ordering', 1]]

    skip_include = [['decomposition', 0], ['pattern', 0], ['quadprec', 0], ['quadstore', 0], ['scaling_thresh', -1], ['mtype', 'double'], ['ordering', 0]]
    skip_pattern_include = [['decomposition', 0], ['pattern', 1], ['quadprec', 0], ['quadstore', 0], ['scaling_thresh', -1], ['ordering', 0]]
    
    skip_order_include = [['decomposition', 0], ['pattern', 0], ['quadprec', 0], ['quadstore', 0], ['scaling_thresh', -1], ['mtype', 'double'], ['ordering', 2]]
    skip_order_pattern_include = [['decomposition', 0], ['pattern', 1], ['quadprec', 0], ['quadstore', 0], ['scaling_thresh', -1], ['ordering', 2]]

    


    parallel_df = prepare_dataset(df, parallel_include, slice_include, exclude)[0]
    parallel_pattern_df = prepare_dataset(df, parallel_pattern_include, slice_include, exclude)[0]
    
    sparse_df = prepare_dataset(df, sparse_include, slice_include, exclude)[0]
    sparse_pattern_df = prepare_dataset(df, sparse_pattern_include, slice_include, exclude)[0]
    sparse_order_df = prepare_dataset(df, sparse_order_include, slice_include, exclude)[0]
    sparse_order_pattern_df = prepare_dataset(df, sparse_order_pattern_include, slice_include, exclude)[0]

    sparse_order_df['algo_name'] = sparse_order_df['algo_name'] + ' SparseOrder'
    sparse_order_pattern_df['algo_name'] = sparse_order_pattern_df['algo_name'] + ' SparseOrder'

    
    skip_df = prepare_dataset(df, skip_include, slice_include, exclude)[0]
    skip_pattern_df = prepare_dataset(df, skip_pattern_include, slice_include, exclude)[0]
    skip_order_df = prepare_dataset(df, skip_order_include, slice_include, exclude)[0]
    skip_order_pattern_df = prepare_dataset(df, skip_order_pattern_include, slice_include, exclude)[0]

    skip_order_df['algo_name'] = skip_order_df['algo_name'] + ' SkipOrder'
    skip_order_pattern_df['algo_name'] = skip_order_pattern_df['algo_name'] + ' SkipOrder'

    skip_order_df.to_csv("whynotskip.csv")
    skip_order_pattern_df.to_csv("whynotskip2.csv")

    df = pd.concat([parallel_df,
                    parallel_pattern_df,
                    sparse_df,
                    sparse_pattern_df,
                    sparse_order_df,
                    sparse_order_pattern_df,
                    skip_df,
                    skip_pattern_df,
                    skip_order_df,
                    skip_order_pattern_df])


    include = [['decomposition', 0], ['quadprec', 0], ['quadstore', 0], ['scaling_thresh', -1], ['mtype', 'double'], ['pattern', 0], ['threads', 1]]

    pattern_include = [['decomposition', 0], ['quadprec', 0], ['quadstore', 0], ['scaling_thresh', -1], ['pattern', 1], ['threads', 1]]
    
    cpus_graph(df, 'synth_size', 'time', 'algo_name', 'synth_density', densities,
                  include, slice_include, exclude, 'real')

    cpus_graph(df, 'synth_size', 'time', 'algo_name', 'synth_density', densities,
                  pattern_include, slice_include, exclude, 'pattern')

    #general_graph(df, 'synth_size', 'time', 'algo_name', 'synth_density', densities,
                  #pattern_include, slice_include, exclude, 'pattern')

                      
            
    ##ALWAYS AT THE END
    general_graph(df, 'synth_size', 'time', 'algo_name', 'synth_density', densities,
                  include, slice_include, exclude, 'THIS IS EMPTY')
    ##ALWAYS AT THE END
    
    plt.tight_layout()
    plt.show()

    print('type: ', df.mtype.unique())
    print('patterns: ', df.pattern.unique())
    print('realw_name: ', df.realw_name.unique())
    print('algo_name:', df.algo_name.unique())
    print('ordering:', df.ordering.unique())
    print('densities:', df.synth_density.unique())
    print('decomposition:', df.decomposition.unique())
    print('calculation half-precision:', df.halfprec.unique())
    print('calculation quad-precision:', df.quadprec.unique())
    print('storage half-precision:', df.halfstore.unique())
    print('storage quad-precision:', df.quadstore.unique())
    print('scaling: ', df.scaling_thresh.unique())
    print('synth_size: ', df.synth_size.unique())
    #df.reindex(columns=column_order(df.algo_name.unique()))
    #print('ordered: ', column_order(df.algo_name.unique()))

    #densities = ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80']
    #general_graph(df, 'synth_size', 'time', 'gpu_num', 'synth_density', densities)
    #plt.tight_layout()
    #plt.show()

    #print(df.shape())
    #df = df.loc[(df['ordering'] == 2)]
    #print(df.head(50))
    df.to_csv("why.csv")
    
    
    
    
