import os
import sys
import glob

def scrape_command(command):

    scraped = command[2:]
    scraped = scraped.replace(" ", "_")
    scraped = scraped.replace("/", "_")

    return scraped

def check_outputs():
    
    done_arr = []

    files = glob.glob('*.stdtxt')
    print('Files: ', files)

    for f in files:
        reader = open(f, 'r')
        lines = reader.readlines()
        command = lines[0].strip('**command: ').strip('\n')
        command = command.replace(' ', '')
        counter = 0
        for l in lines:
            if(l.find('Result') != -1):
                counter += 1

        reader.close()
        if(counter >= 5):
            print('Command: ', command, '->done!')
            done_arr.append(command)
        
            
        else:
            errfile = f.strip('.stdtxt') + 'errtxt'
            print('Command', command, '-> not done!')
            print('Removing: ', f)
            try:
                os.remove(f)
            except:
                print('Exception: File already removed!')

            print('Removing: ', errfile)
            try:
                os.remove(errfile)
            except:
                print('Exception: File already removed!')
            

        

    return done_arr
    

def check_done(command, done_arr):
    if command in done_arr:
        print('True')
        return True
    else:
        print('False')
        return False


if __name__ == "__main__":

    commands_file = sys.argv[1]
    reader = open(commands_file, 'r')
    lines = reader.readlines()
    reader.close()
    #print(lines)                                                                                                                                                                                          

    done_arr = check_outputs()
    #print(done_arr)

    scrl = []
    counter = 0;

    for line in lines:
        to_write = ''
        command = line.strip('\n')
        print('Checking for command: ', command)
        if(check_done(command.replace(' ', ''), done_arr)):
            print('Skipping: ', command)
            continue
        to_write += '####' + command + '####' + '\n'
        out_scraped = scrape_command(command)
        out_name = 'out_' + str(counter) + '_' + out_scraped
        to_write += out_name + '=$(' + command + ' 1>' + ' ' + out_name + '.stdtxt' + ' 2> ' + ' ' + out_name + '.errtxt' + ')'+ ' \n'
        to_write += 'wait \n'
        scrl.append(to_write)

        counter += 1


    writer = open(commands_file.strip('.txt') + '_scripts.sh', 'w+')
    for item in scrl:
        writer.write(item)

    writer.close()
