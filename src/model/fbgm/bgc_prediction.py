import pandas as pd
import numpy as np
import torch as th
import torch.nn as nn
from . import model
import json
import time

def performBGCPrediction(file_contig_list,file_pfam_json,path_esm_json,path_genome_mining_result,ref_file_pfam_A_dat,ofile_df_putative_BGC,min_n_pep_contig,min_n_pep_BGC,max_n_pep_BGC,max_interval_n_pep_BGC_merging,pred_score_threshold,pred_score_top_ratio,dict_pred_score_threshold_core_enzyme,path_model,n_submodel,device):

    print("Performing BGC prediction...")
    print()
    s_time=time.time()

    dict_pfam_description={}
    with open(ref_file_pfam_A_dat,'r') as f:
        pfam_id,pfam_description=None,None
        for line in f.readlines():
            line=line.strip()
            if line.startswith("#=GF ID"):
                pfam_description=line[7:].strip()
            if line.startswith("#=GF AC"):
                pfam_id=line[7:].strip()
                dict_pfam_description[pfam_id]=pfam_description

    with open(file_pfam_json,'r') as f:
        dict_pep2pfam=json.load(f)
    dict_pep2pfam={pep_id:[(pfam_id,start,end,relative_start,relative_end) for pfam_id,start,end,relative_start,relative_end,pfam_seq in dict_pep2pfam[pep_id]] for pep_id in dict_pep2pfam}
    dict_pep2pfam_description={pep_id:[dict_pfam_description[pfam_id] for pfam_id,start,end,relative_start,relative_end in dict_pep2pfam[pep_id]] for pep_id in dict_pep2pfam}

    max_interval_n_pep_BGC_merging_=max_interval_n_pep_BGC_merging+1

    l_contig=np.load(file_contig_list,allow_pickle=True)
    dict_esm={}
    dict_df_genome_mining_result={}
    dict_pep2pred_score={}
    l_all_pred_score=[]
    for contig in l_contig:
        file_esm_json=path_esm_json+contig+".ESM.json"
        with open(file_esm_json,'r') as f:
            dict_esm_=json.load(f)
        dict_esm.update(dict_esm_)
        df_genome_mining_result=pd.read_csv(path_genome_mining_result+contig+".csv")
        l_pep_id=df_genome_mining_result["pep_id"].values
        l_pred_score=df_genome_mining_result["pred_score"].values
        n_pep=len(df_genome_mining_result)
        if n_pep<min_n_pep_contig:
            continue
        dict_df_genome_mining_result[contig]=df_genome_mining_result
        dict_pep2pred_score.update(dict(zip(l_pep_id,l_pred_score)))
        l_all_pred_score+=list(l_pred_score)
    if pred_score_threshold is None:
        pred_score_threshold=np.quantile(l_all_pred_score,(1-pred_score_top_ratio))

    a=[]
    l_BGC_id=[]
    dict_BGC_id2l_pep_id={}
    no_putative_BGC=0
    for contig in l_contig:

        if contig not in dict_df_genome_mining_result:
            continue
        df_genome_mining_result=dict_df_genome_mining_result[contig]
        
        dict_pep_info={}
        for i in range(len(df_genome_mining_result)):
            pep_id,start,end=df_genome_mining_result["pep_id"][i],df_genome_mining_result["start"][i],df_genome_mining_result["end"][i]
            dict_pep_info[pep_id]={"index":i,"start":start,"end":end}
        l_pep_id=df_genome_mining_result["pep_id"].values

        flag_recording=False
        l_basic_gene_cluster_range=[]
        basic_gene_cluster=[]
        for i in range(len(df_genome_mining_result)):
            pep_id,pred_score=df_genome_mining_result["pep_id"][i],df_genome_mining_result["pred_score"][i]
            if pred_score>=pred_score_threshold:
                if not flag_recording:
                    flag_recording=True
                basic_gene_cluster.append(pep_id)
            else:
                if flag_recording:
                    l_basic_gene_cluster_range.append({"index_start_pep":dict_pep_info[basic_gene_cluster[0]]["index"],"index_end_pep":dict_pep_info[basic_gene_cluster[-1]]["index"]})
                    flag_recording=False
                    basic_gene_cluster=[]
        if flag_recording:
            l_basic_gene_cluster_range.append({"index_start_pep":dict_pep_info[basic_gene_cluster[0]]["index"],"index_end_pep":dict_pep_info[basic_gene_cluster[-1]]["index"]})

        l_merged_gene_cluster_range=[]
        last_index_basic_gene_cluster=len(l_basic_gene_cluster_range)-1
        for i in range(len(l_basic_gene_cluster_range)):
            current_basic_gene_cluster_range=l_basic_gene_cluster_range[i]
            if i==last_index_basic_gene_cluster:
                l_merged_gene_cluster_range.append(current_basic_gene_cluster_range)
                continue

            next_basic_gene_cluster_range=l_basic_gene_cluster_range[i+1]
            current_index_start_pep,current_index_end_pep=current_basic_gene_cluster_range["index_start_pep"],current_basic_gene_cluster_range["index_end_pep"]
            next_index_start_pep,next_index_end_pep=next_basic_gene_cluster_range["index_start_pep"],next_basic_gene_cluster_range["index_end_pep"]
            if next_index_start_pep-current_index_end_pep<=max_interval_n_pep_BGC_merging_:
                l_basic_gene_cluster_range[i+1]={"index_start_pep":current_index_start_pep,"index_end_pep":next_index_end_pep}
            else:
                l_merged_gene_cluster_range.append(current_basic_gene_cluster_range)

        
        if len(l_merged_gene_cluster_range)>0:

            for merged_gene_cluster_range in l_merged_gene_cluster_range:
                b=[]
                index_start_pep,index_end_pep=merged_gene_cluster_range["index_start_pep"],merged_gene_cluster_range["index_end_pep"]
                for index in range(index_start_pep,index_end_pep+1):
                    pep_id=l_pep_id[index]
                    pep_info=dict_pep_info[pep_id]
                    start,end=pep_info["start"],pep_info["end"]
                    b.append([pep_id,start,end])
                df_merged_gene_cluster=pd.DataFrame(b,columns=["pep_id","start","end"])
                tl_pep_id=list(df_merged_gene_cluster["pep_id"].values)
                n_pep=len(tl_pep_id)
                if n_pep>=min_n_pep_BGC and n_pep<=max_n_pep_BGC:
                    tmp_dict_pep2pfam_description={}
                    for pep_id in tl_pep_id:
                        tl_pfam_description=dict_pep2pfam_description[pep_id]
                        if len(tl_pfam_description)>0:
                            tmp_dict_pep2pfam_description[pep_id]=tl_pfam_description
                    no_putative_BGC+=1
                    BGC_id="BGC_"+str(no_putative_BGC)
                    l_BGC_id.append(BGC_id)
                    dict_BGC_id2l_pep_id[BGC_id]=tl_pep_id
                    tl_pred_score=[dict_pep2pred_score[pep_id] for pep_id in tl_pep_id]
                    a.append([BGC_id,contig,df_merged_gene_cluster["start"].min(),df_merged_gene_cluster["end"].max(),n_pep,str(tl_pep_id),str(tmp_dict_pep2pfam_description),np.mean(tl_pred_score).round(4),np.max(tl_pred_score).round(4)])

    df_putative_BGC=pd.DataFrame(a,columns=["BGC_id","contig","start","end","#pep","pep_id","pfam_description","confidence_score (mean)","confidence_score (max)"])

    file_core_enzyme_identification_model_config_json=path_model+"core_enzyme_identification_model_config.json"
    with open(file_core_enzyme_identification_model_config_json) as f:
        dict_core_enzyme_identification_model_config=json.load(f)
    d_raw_predefined_appendix=dict_core_enzyme_identification_model_config["d_raw_predefined_appendix"]
    d_raw_extended_appendix=dict_core_enzyme_identification_model_config["d_raw_extended_appendix"]

    file_pfam2index_json=path_model+"pfam2index.json"
    with open(file_pfam2index_json,'r') as f:
        dict_pfam2index=json.load(f)
    l_pfam_id=[None]*len(dict_pfam2index)
    for pfam_id in dict_pfam2index:
        l_pfam_id[dict_pfam2index[pfam_id]]=pfam_id

    core_enzyme_identification_model=model.CoreEnzymeIdentificationModel(
        l_pfam_id=l_pfam_id,
        d_pfam=dict_core_enzyme_identification_model_config["d_pfam"],
        d_raw_predefined_appendix=dict_core_enzyme_identification_model_config["d_raw_predefined_appendix"],
        d_processed_predefined_appendix=dict_core_enzyme_identification_model_config["d_processed_predefined_appendix"],
        d_raw_extended_appendix=dict_core_enzyme_identification_model_config["d_raw_extended_appendix"],
        d_processed_extended_appendix=dict_core_enzyme_identification_model_config["d_processed_extended_appendix"],
        d_esm=dict_core_enzyme_identification_model_config["d_esm"],
        n_head=dict_core_enzyme_identification_model_config["n_head"],
        d_FFN=dict_core_enzyme_identification_model_config["d_FFN"],
        n_TFE_module_pfam_level=dict_core_enzyme_identification_model_config["n_TFE_module_pfam_level"],
        n_TFE_layer_pfam2pep_level=dict_core_enzyme_identification_model_config["n_TFE_layer_pfam2pep_level"],
        n_TFE_layer_pep_level=dict_core_enzyme_identification_model_config["n_TFE_layer_pep_level"],
        n_layer_MLP=dict_core_enzyme_identification_model_config["n_layer_MLP"],
        activation_func=eval(dict_core_enzyme_identification_model_config["activation_func"]),
        dropout_rate=dict_core_enzyme_identification_model_config["dropout_rate"]
    ).to(device)
    
    dict_BGC_id2pred_score_core_enzyme={}
    file_core_enzyme_list=path_model+"core_enzyme_list.npy"
    l_core_enzyme=np.load(file_core_enzyme_list,allow_pickle=True)
    for core_enzyme in l_core_enzyme:
        for no_submodel in range(n_submodel):
            path_model_=path_model+str(no_submodel)+"/core_enzyme_identification_model/"+core_enzyme+'/'
            file_core_enzyme_identification_model=path_model_+"model.pkl"
            file_extended_appendix_json=path_model_+"extended_appendix.json"
            with open(file_extended_appendix_json,'r') as f:
                dict_extended_appendix=json.load(f)      
            core_enzyme_identification_model.load_state_dict(th.load(file_core_enzyme_identification_model,map_location=device))
            core_enzyme_identification_model.eval()

            for BGC_id in l_BGC_id:
                tl_pep_id=dict_BGC_id2l_pep_id[BGC_id]
                dict_data=model.genDataDict(
                    l_pep_id=tl_pep_id,d_raw_predefined_appendix=d_raw_predefined_appendix,d_raw_extended_appendix=d_raw_extended_appendix,dict_pep2pfam=dict_pep2pfam,dict_esm=dict_esm,dict_extended_appendix=dict_extended_appendix,token_prefix="[PRE]",token_null="[NULL]"
                )
                with th.no_grad():
                    tl_pred_score_core_enzyme=core_enzyme_identification_model([dict_data],need_weight=False)[0][0].cpu().detach().numpy()
                if core_enzyme not in dict_BGC_id2pred_score_core_enzyme:
                    dict_BGC_id2pred_score_core_enzyme[core_enzyme]={}
                if BGC_id not in dict_BGC_id2pred_score_core_enzyme[core_enzyme]:
                    dict_BGC_id2pred_score_core_enzyme[core_enzyme][BGC_id]=tl_pred_score_core_enzyme
                else:
                    dict_BGC_id2pred_score_core_enzyme[core_enzyme][BGC_id]+=tl_pred_score_core_enzyme

    dict_BGC_id2dict_core_enzyme={}  
    for BGC_id in l_BGC_id:
        tl_pep_id=dict_BGC_id2l_pep_id[BGC_id]
        if BGC_id not in dict_BGC_id2dict_core_enzyme:
            dict_BGC_id2dict_core_enzyme[BGC_id]={pep_id:[] for pep_id in tl_pep_id}
        for core_enzyme in l_core_enzyme:
            tl_pred_score_core_enzyme=dict_BGC_id2pred_score_core_enzyme[core_enzyme][BGC_id]/n_submodel
            pred_score_threshold_core_enzyme=dict_pred_score_threshold_core_enzyme[core_enzyme]
            for pep_id,pred_score_core_enzyme in zip(tl_pep_id,tl_pred_score_core_enzyme):
                if pred_score_core_enzyme>=pred_score_threshold_core_enzyme:
                    dict_BGC_id2dict_core_enzyme[BGC_id][pep_id].append(core_enzyme)
    dict_BGC_id2str_dict_core_enzyme={}
    for BGC_id in dict_BGC_id2dict_core_enzyme:
        tl_pep_id=dict_BGC_id2l_pep_id[BGC_id]
        tmp_dict_core_enzyme=dict_BGC_id2dict_core_enzyme[BGC_id]
        tmp_dict_core_enzyme_={}
        for pep_id in tl_pep_id:
            tl_core_enzyme=tmp_dict_core_enzyme[pep_id]
            if len(tl_core_enzyme)>0:
                tmp_dict_core_enzyme_[pep_id]=tl_core_enzyme
        dict_BGC_id2str_dict_core_enzyme[BGC_id]=str(tmp_dict_core_enzyme_)
    df_putative_BGC["putative_core_enzyme"]=df_putative_BGC["BGC_id"].map(dict_BGC_id2str_dict_core_enzyme)
    df_putative_BGC=df_putative_BGC[["BGC_id","contig","start","end","#pep","pep_id","putative_core_enzyme","pfam_description","confidence_score (mean)","confidence_score (max)"]].copy()

    e_time=time.time()
    n_putative_BGC=len(df_putative_BGC)
    if n_putative_BGC==0:
        raise Exception(
            "No BGC identified. Please try to adjust the prediction score threshold for more possible results."
        )
    else:
        df_putative_BGC.to_csv(ofile_df_putative_BGC,index=None)
        print("Finished, {:d}s taken. Identify {:d} BGC(s) in total. The BGC prediction result is saved as {:s}.".format(int(e_time-s_time),n_putative_BGC,ofile_df_putative_BGC))
        print()

    return pred_score_threshold