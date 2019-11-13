from utils.download import download_file_from_google_drive
import os

if __name__ == '__main__':
    id = '14ZCz3l11ehCl8_E1P0YSbF__PK4SwcBZ'
    name = "temp.zip"
    os.chdir('../')
    if not os.path.isdir('./tmp'):
        os.makedirs('./tmp')
    os.chdir('tmp')
    print('Start Download')
    download_file_from_google_drive(id, name)
    print('Finish Download')
    os.system('unzip temp.zip')
    if not os.path.isdir('../configs'):
        os.makedirs('../configs')
    if not os.path.isdir('../results'):
        os.makedirs('../results')
    os.chdir('./models/configs')
    os.system('mv * ../../../configs/')
    os.chdir('../results')
    os.system('mv * ../../../results/')

