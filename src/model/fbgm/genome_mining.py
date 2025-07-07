import pandas as pd
import numpy as np
import torch as th
import torch.nn as nn
from . import model
from Bio import SeqIO
import json
import time

def performGenomeMining(file_contig_list,path_seq,file_pfam_json,path_esm_json,file_df_pep_seq,opath_genome_mining_result,path_model,n_submodel,device):

    print("Running f-BGM...")
    print()
    s_time=time.time()

    with open(file_pfam_json,'r') as f:
        dict_pep2pfam=json.load(f)
    dict_pep2pfam={pep_id:[(pfam_id,start,end,relative_start,relative_end) for pfam_id,start,end,relative_start,relative_end,pfam_seq in dict_pep2pfam[pep_id]] for pep_id in dict_pep2pfam}

    dict_contig2l_pep_id={}
    dict_esm={}
    df_pep_seq=pd.read_csv(file_df_pep_seq)
    grouped_df_pep_seq=df_pep_seq.groupby("contig")
    l_contig=np.load(file_contig_list,allow_pickle=True)
    for contig in l_contig:
        file_fasta=path_seq+contig+".fasta"
        l_pep_id=[record.id for record in SeqIO.parse(file_fasta,"fasta")]
        dict_contig2l_pep_id[contig]=l_pep_id
        file_esm_json=path_esm_json+contig+".esm.json"
        with open(file_esm_json,'r') as f:
            dict_esm_=json.load(f)
        dict_esm.update(dict_esm_)

    file_genome_mining_model_config_json=path_model+"genome_mining_model_config.json"
    with open(file_genome_mining_model_config_json) as f:
        dict_genome_mining_model_config=json.load(f)
    d_raw_predefined_appendix=dict_genome_mining_model_config["d_raw_predefined_appendix"]
    d_raw_extended_appendix=dict_genome_mining_model_config["d_raw_extended_appendix"]

    file_pfam2index_json=path_model+"pfam2index.json"
    with open(file_pfam2index_json,'r') as f:
        dict_pfam2index=json.load(f)
    l_pfam_id=[None]*len(dict_pfam2index)
    for pfam_id in dict_pfam2index:
        l_pfam_id[dict_pfam2index[pfam_id]]=pfam_id

    genome_mining_model=model.GenomeMiningModel(
        l_pfam_id=l_pfam_id,
        l_genomic_window_size=dict_genome_mining_model_config["l_genomic_window_size"],
        d_pfam=dict_genome_mining_model_config["d_pfam"],
        d_raw_predefined_appendix=dict_genome_mining_model_config["d_raw_predefined_appendix"],
        d_processed_predefined_appendix=dict_genome_mining_model_config["d_processed_predefined_appendix"],
        d_raw_extended_appendix=dict_genome_mining_model_config["d_raw_extended_appendix"],
        d_processed_extended_appendix=dict_genome_mining_model_config["d_processed_extended_appendix"],
        d_esm=dict_genome_mining_model_config["d_esm"],
        d_hidden_LSTM=dict_genome_mining_model_config["d_hidden_LSTM"],
        n_head=dict_genome_mining_model_config["n_head"],
        d_FFN=dict_genome_mining_model_config["d_FFN"],
        n_TFE_module_pfam_level=dict_genome_mining_model_config["n_TFE_module_pfam_level"],
        n_TFE_layer_pfam2pep_level=dict_genome_mining_model_config["n_TFE_layer_pfam2pep_level"],
        n_TFE_layer_pep_level=dict_genome_mining_model_config["n_TFE_layer_pep_level"],
        n_layer_LSTM=dict_genome_mining_model_config["n_layer_LSTM"],
        n_layer_MLP=dict_genome_mining_model_config["n_layer_MLP"],
        activation_func=eval(dict_genome_mining_model_config["activation_func"]),
        dropout_rate=dict_genome_mining_model_config["dropout_rate"]
    ).to(device)

    dict_contig2l_pred_score={}
    for no_submodel in range(n_submodel):
        path_model_=path_model+str(no_submodel)+"/genome_mining_model/"
        file_genome_mining_model=path_model_+"model.pkl"
        file_extended_appendix_json=path_model_+"extended_appendix.json"
        with open(file_extended_appendix_json,'r') as f:
            dict_extended_appendix=json.load(f)      
        genome_mining_model.load_state_dict(th.load(file_genome_mining_model,map_location=device))
        genome_mining_model.eval()

        for contig in l_contig:
            l_pep_id=dict_contig2l_pep_id[contig]
            dict_data=model.genDataDict(
                l_pep_id=l_pep_id,d_raw_predefined_appendix=d_raw_predefined_appendix,d_raw_extended_appendix=d_raw_extended_appendix,dict_pep2pfam=dict_pep2pfam,dict_esm=dict_esm,dict_extended_appendix=dict_extended_appendix,token_prefix="[PRE]",token_null="[NULL]"
            )
            with th.no_grad():
                l_pred_score=genome_mining_model([dict_data],need_weight=False)[0][0].cpu().detach().numpy()
            if contig not in dict_contig2l_pred_score:
                dict_contig2l_pred_score[contig]=l_pred_score
            else:
                dict_contig2l_pred_score[contig]+=l_pred_score

    for contig in l_contig:
        l_pep_id=dict_contig2l_pep_id[contig]
        l_pred_score=list(dict_contig2l_pred_score[contig]/n_submodel)
        dict_pred_score=dict(zip(l_pep_id,l_pred_score))

        tdf_pep_seq=grouped_df_pep_seq.get_group(contig).reset_index(drop=True).copy()
        tdf_pep_seq["pred_score"]=tdf_pep_seq["pep_id"].map(dict_pred_score)
        df_genome_mining_result=tdf_pep_seq[["gene_id","pep_id","contig","start","end","pred_score"]].copy()
        df_genome_mining_result.to_csv(opath_genome_mining_result+contig+".csv",index=None)

    e_time=time.time()
    print("Finished, {:d}s taken.".format(int(e_time-s_time)))
    print()
        