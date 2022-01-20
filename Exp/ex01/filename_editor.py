# %%
from PIL import Image
import os
import glob 
import zipfile
import shutil

# %%
dir = os.getenv('HOME')+'/DL/aiffel/Exp/ex01/rock_scissor_paper_dataset/'

# %%
zip_dir = dir + 'zip_archive/'
rock_dir = dir + 'rock'
paper_dir = dir + 'paper'
scissor_dir = dir + 'scissor'

stack = 0
file_list = ['paper', 'scissor', 'rock']
dest_dir = [paper_dir,scissor_dir,rock_dir]

# %%
zip_list = glob.glob(zip_dir+'*')
for i, _ in enumerate(zip_list):
    zip_list[i] = zip_list[i] + '/'
    
# zip_list

# %%
def unzip_file(dir):
    
    for i in file_list:
            
        try:
            if not os.path.exists(dir + i):
                os.makedirs(dir + i)
                
                tmp_zip = zipfile.ZipFile(dir + i + '.zip')
                tmp_zip.extractall(dir + i)
        except:
            continue
        

# %%
for i in zip_list:
    unzip_file(i)

# %%
for i in zip_list:
    for s, d in zip(file_list, dest_dir):
        source = i + s + '/'
        dest = d
        print(source)
        print(dest)
        files = os.listdir(source)
        for file in files:
            new_file_name = str(int(file.split('.')[0])+stack)+'.jpg'
            os.rename(source+ file,source + new_file_name)
            shutil.move(f'{source}/{new_file_name}',dest,copy_function= shutil.copytree)
    stack += 100

# %%



