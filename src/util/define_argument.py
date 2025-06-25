import argparse
import re

def define_numeric_range(numeric_type,minimum,maximum):
    def define_numeric_range_(value):
        try:
            value_=eval(numeric_type+'('+value+')')
        except:
            msg="Invalid value: {:s} value expected".format(numeric_type)
            raise argparse.ArgumentTypeError(msg)
        
        if value_<minimum or value_>maximum:
            if numeric_type=="float":
                msg="Invalid value: {:s} value in [{:.2f},{:.2f}] expected".format(numeric_type,minimum,maximum)
            elif numeric_type=="int":
                msg="Invalid value: {:s} value in [{:d},{:d}] expected".format(numeric_type,minimum,maximum)
            raise argparse.ArgumentTypeError(msg)
        return value_
    
    return define_numeric_range_


def define_key_value_pairs():
    def define_key_value_pairs_(s):
        try:
            regex="([\w|\-]+)\=([\w|\.]+)"
            l_kv=re.findall(regex,s)
            if len(l_kv)==0:
                raise Exception()
            d={}
            for k,v in l_kv:
                d[k]=float(v)
        except:
            msg='Invalid format: format like "key1=value1 key2=value2 key3=value3 ..." expected'
            raise argparse.ArgumentTypeError(msg)
        return d
    
    return define_key_value_pairs_