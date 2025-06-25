import torch as th
import psutil

def getAvailableDevice(required_memory,min_retained_gpu_memory,min_retained_system_memory,device_type=None):
    if device_type=="gpu":
        if th.cuda.is_available():
            n_gpu=th.cuda.device_count()
            for i in range(n_gpu):
                free_gpu_memory,total_gpu_memory=th.cuda.mem_get_info(i)
                if free_gpu_memory>=min_retained_gpu_memory+required_memory:
                    th.cuda.set_device(i)
                    return ("gpu",i,th.device("cuda:"+str(i)))
        else:
            return None
        
    elif device_type=="cpu":
        system_memory_info=psutil.virtual_memory()
        free_system_memory=system_memory_info.available
        if free_system_memory>=min_retained_system_memory+required_memory:
            return ("cpu",None,th.device("cpu"))
        else:
            return None
        
    else:
        if th.cuda.is_available():
            n_gpu=th.cuda.device_count()
            for i in range(n_gpu):
                free_gpu_memory,total_gpu_memory=th.cuda.mem_get_info(i)
                if free_gpu_memory>=min_retained_gpu_memory+required_memory:
                    th.cuda.set_device(i)
                    return ("gpu",i,th.device("cuda:"+str(i)))

        system_memory_info=psutil.virtual_memory()
        free_system_memory=system_memory_info.available
        if free_system_memory>=min_retained_system_memory+required_memory:
            return ("cpu",None,th.device("cpu"))
        else:
            return None