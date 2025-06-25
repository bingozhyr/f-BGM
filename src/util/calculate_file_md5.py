import hashlib
 
def calFileMD5(file,block_size=4096):
    md5=hashlib.md5()
    with open(file,"rb") as f:
        while True:
            data=f.read(block_size)
            if not data:
                break        
            md5.update(data)     
    return md5.hexdigest()