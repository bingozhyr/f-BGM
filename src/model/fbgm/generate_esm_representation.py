import numpy as np
import torch as th
from . import model
from Bio import SeqIO
import json
import time


def getSubSeqList(seq,max_sub_seq_length,expected_sub_seq_coverage,sub_seq_coverage_decay_step,min_sub_seq_coverage,expected_max_n_sub_seq):
    if len(seq)<=max_sub_seq_length:
        return [seq]
    else:
        while True:
            step=max(int(max_sub_seq_length*(1-expected_sub_seq_coverage)),1)
            if (step*(expected_max_n_sub_seq-1)+max_sub_seq_length)<len(seq):
                if expected_sub_seq_coverage!=min_sub_seq_coverage:
                    expected_sub_seq_coverage-=sub_seq_coverage_decay_step
                    if expected_sub_seq_coverage<min_sub_seq_coverage:
                        expected_sub_seq_coverage=min_sub_seq_coverage
                    continue
            l_sub_seq=[]
            start_pos=0
            end_pos=max_sub_seq_length
            while True:
                sub_seq=seq[start_pos:end_pos]
                l_sub_seq.append(sub_seq)
                if end_pos>=len(seq):
                    break
                start_pos+=step
                end_pos+=step
            return l_sub_seq
    
    
def genESMRepresentation(file_contig_list,path_seq,opath_esm_json,path_model,device):

    print("Generating ESM representation...")
    print()
    s_time=time.time()

    file_esm_model_config_json=path_model+"esm_config.json"
    with open(file_esm_model_config_json) as f:
        dict_esm_model_config=json.load(f)
    max_sub_seq_length=dict_esm_model_config["max_sub_seq_length"]
    expected_sub_seq_coverage=dict_esm_model_config["expected_sub_seq_coverage"]
    sub_seq_coverage_decay_step=dict_esm_model_config["sub_seq_coverage_decay_step"]
    min_sub_seq_coverage=dict_esm_model_config["min_sub_seq_coverage"]
    expected_max_n_sub_seq=dict_esm_model_config["expected_max_n_sub_seq"]
    
    esm_model=model.ESMModel(
        d_encoding=dict_esm_model_config["d_encoding"],
        pretrained_model_name=dict_esm_model_config["pretrained_model_name"],
    ).to(device)
    file_esm_model=path_model+"esm.pkl"
    esm_model.load_state_dict(th.load(file_esm_model,map_location=device))
    esm_model.eval()

    l_contig=np.load(file_contig_list,allow_pickle=True)
    for contig in l_contig:
        file_fasta_contig_pep_seq=path_seq+contig+".fasta"
        l_pep_id=[]
        dict_pep_seq={}
        for record in SeqIO.parse(file_fasta_contig_pep_seq,"fasta"):
            record_id,record_seq=record.id,str(record.seq)
            l_pep_id.append(record_id)
            dict_pep_seq[record_id]=record_seq
        odict={}
        for pep_id in l_pep_id:
            pep_seq=dict_pep_seq[pep_id]
            with th.no_grad():
                odict[pep_id]=th.concat(
                    [esm_model([sub_seq]) for sub_seq in getSubSeqList(pep_seq,max_sub_seq_length,expected_sub_seq_coverage,sub_seq_coverage_decay_step,min_sub_seq_coverage,expected_max_n_sub_seq)]
                ).mean(axis=0).cpu().detach_().numpy().tolist()
        with open(opath_esm_json+contig+".ESM.json",'w') as f:
            json.dump(odict,f)
    
    e_time=time.time()
    print("Finished, {:d}s taken.".format(int(e_time-s_time)))
    print()