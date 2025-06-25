import subprocess
import re
import os

def getPathBaseEnv():
    reg_base_environment="base environment : ((\/[^\/\s]+)+)"
    result=subprocess.run("conda info",shell=True,capture_output=True,text=True)
    if result.returncode!=0:
        return None
    for line in result.stdout.split('\n'):
        reg_result=re.search(reg_base_environment,line)
        if reg_result is not None:
            return os.path.abspath(reg_result.groups()[0])+'/'
    return None

def getCurrentEnvName():
    reg_current_environment="active environment : (\S+)"
    result=subprocess.run("conda info",shell=True,capture_output=True,text=True)
    if result.returncode!=0:
        return None
    for line in result.stdout.split('\n'):
        reg_result=re.search(reg_current_environment,line)
        if reg_result is not None:
            return reg_result.groups()[0]
    return None
    
    
def checkEnvAugustus(path_base_env,env_name):
    l_cmd=['source '+path_base_env+'etc/profile.d/conda.sh','conda activate '+env_name,"augustus --help"]
    cmd="bash -c '"+" && ".join(l_cmd)+"'"
    
    print("Checking the AUGUSTUS environment...")
    print()
    result=subprocess.run(cmd,shell=True,capture_output=True,text=True)
    if result.returncode==0:
        print("Finished.")
        print()
    else:
        raise Exception(
            "Failed to locate the AUGUSTUS program."
        )