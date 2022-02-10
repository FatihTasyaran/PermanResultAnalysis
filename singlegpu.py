def gen_file_names(sizes, densities, postfixes, paths):

    files = []
    
    for path in paths:
        for size in sizes:
            for density in densities:
                for postfix in postfixes:
                    one_file = path + size + '_' + density + '_' + postfix + '.mtx'
                    files.append(one_file)

    return files


def gen_dense_commands(commands, base_str, files, dense_algos, pattern, real_dtype, int_dtype):

    dtype = []
    order = ['', ' -r1']

    for filen in files:
        
        if(filen.find('erdos_int') != -1):
            pattern = ['']
            dtype = int_dtype
        else:
            pattern = ['', ' -b']
            dtype = real_dtype
            
        for algo in dense_algos:
            for pat in pattern:
                
                if(pat == ' -b'):
                    dtype = int_dtype
                    
                    for dt in dtype:

                        for ordr in order:
                            command = base_str + ' -f '  + filen + algo + pat + dt + ordr
                            commands.append(command)
    
    return commands
                            

def gen_sparse_commands(commands, base_str, files, sparse_algos, pattern, real_dtype, int_dtype):
     
    dtype = []
    order = []
    sort_order = ['', ' -r1']
    skip_order = ['', ' -r2']
    
    for filen in files:
        
        if(filen.find('erdos_int') != -1):
            pattern = ['']
            dtype = int_dtype
        elif(filen.find('erdos_real') != -1):
            dtype = real_dtype
            pattern = ['', ' -b']
        else:
            print('weird!', filen)
            exit

        print(filen, dtype, pattern)
        for algo in sparse_algos:
            if(algo == ' -s -p14'):
                order = skip_order
            else:
                order = sort_order
            for pat in pattern:
                
                if(pat == ' -b'):
                    dtype = int_dtype
                    
                    
                    for dt in dtype:

                        for ordr in order:
                            command = base_str + ' -f '  + filen + algo + pat + dt + ordr
                            commands.append(command)

    return commands

if __name__ == "__main__":
    
    base_str = "./gpu_perman -k6"

    
    paths = ['erdos_int/', 'erdos_real/'] ##And then once more for the real matrices
    

    #sizes = ['30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']
    sizes = ['30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']
    densities = ['0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80']
    postfixes = ['3']
    
    files = gen_file_names(sizes, densities, postfixes, paths)
    
    #for item in files:
        #print(item)
    
    dense_algos = [' -p1', ' -p2', ' -p3', ' -p4']
    sparse_algos = [' -s -p1', ' -s -p2', ' -s -p3', ' -s -p4', ' -s -p14']

    weird_algos = [' -p21']

    pattern = ['', ' -b']
    
    real_dtype = ['', ' -h', ' -w', ' -h -w']
    int_dtype = ['', ' -h']

    commands = []
    
    commands = gen_dense_commands(commands, base_str, files, dense_algos, pattern, real_dtype, int_dtype)
    
    #for item in commands:
    #print(item)
    dense_len = len(commands)

    commands = gen_sparse_commands(commands, base_str, files, sparse_algos,  pattern, real_dtype, int_dtype)
    sparse_len = len(commands)
    
    commands = gen_dense_commands(commands, base_str, files, weird_algos, pattern, real_dtype, int_dtype)
    weird_len = len(commands)

    #for item in commands:
    #print(item)
    
    for item in commands:
        print(item)

    print('After synthetic dense commands: ', dense_len)
    print('After synthetic sparse commands: ', sparse_len)
    print('After synthetic weird commands: ', weird_len)
    
    
    writer = open('runs1.txt', 'w+')
    for item in commands:
        writer.write(item + '\n')
    
    
        
