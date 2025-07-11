import pandas as pd
import numpy as np
from Bio import SeqIO,Seq,SeqRecord
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource,HoverTool,CustomJSTickFormatter,Range1d
from bokeh.io import output_file,show
import torch as th
import torch.nn as nn
from ..model.fbgm import model
import json
import plotly.graph_objects as go
import plotly as px
from plotly.offline import plot
import time
l_base_color=px.colors.qualitative.Plotly
n_char_line_breaking_threshold=128

def hex_to_rgba(h,a):
    stripped_h=h.strip('#')
    r,g,b=tuple(int(stripped_h[i:i+2],16) for i in (0,2,4))
    return "rgba({:d},{:d},{:d},{:f})".format(r,g,b,a)


def genPredictionDetail(file_contig_list,file_contig_fasta,file_pfam_json,path_esm_json,path_genome_mining_result,path_prediction_result,opath_prediction_result_BGC_detail,opath_prediction_result_contig_detail,path_model,ref_file_pfam_A_dat,flag_pred_score_threshold,pred_score_threshold,flanking_DNA_sequence_length,n_submodel,device):

    print("Deciphering and visualizing the putative BGC(s) and related contig(s) ...")
    print()
    s_time=time.time()

    dict_record={}
    dict_record_length={}
    for record in SeqIO.parse(file_contig_fasta,"fasta"):
        record_id,record_seq=record.id,record.seq
        dict_record[record_id]=record_seq
        dict_record_length[record_id]=len(record_seq)

    token_null="[NULL]"
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
    dict_pfam_description[token_null]="NULL"

    with open(file_pfam_json,'r') as f:
        dict_pep2pfam=json.load(f)
    dict_pep2pfam={pep_id:[(pfam_id,start,end,relative_start,relative_end) for pfam_id,start,end,relative_start,relative_end,pfam_seq in dict_pep2pfam[pep_id]] for pep_id in dict_pep2pfam}
    dict_pep2pfam_description={pep_id:[dict_pfam_description[pfam_id] for pfam_id,start,end,relative_start,relative_end in dict_pep2pfam[pep_id]] for pep_id in dict_pep2pfam}

    dict_esm={}
    l_contig=np.load(file_contig_list,allow_pickle=True)
    for contig in l_contig:
        file_esm_json=path_esm_json+contig+".esm.json"
        with open(file_esm_json,'r') as f:
            dict_esm_=json.load(f)
        dict_esm.update(dict_esm_)

    l_algorithm_name=["f-BGM"]
    n_algorithm=len(l_algorithm_name)
    l_color=(l_base_color*int(np.ceil(n_algorithm/len(l_base_color))))[:n_algorithm]
    dict_algorithm2color=dict(zip(l_algorithm_name,l_color))
    l_y=list(range(1,n_algorithm+1))[::-1]
    dict_algorithm2y=dict(zip(l_algorithm_name,l_y))

    df_putative_BGC=pd.read_csv(path_prediction_result+"putative_BGC.csv")
    grouped_df_putative_BGC=df_putative_BGC.groupby("contig",sort=False)

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
    
    genome_mining_model_=model.GenomeMiningModel(
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

    genome_mining_model=model.GenomeMiningModel(
        l_pfam_id=l_pfam_id,
        l_genomic_window_size=[np.inf],
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

    print("The following files will be generated for the f-BGM-putative BGC(s) and saved to {:s}:".format(opath_prediction_result_BGC_detail))
    print("1. xxx.dna.fasta ('xxx' is BGC id), which records DNA sequence of corresponding putative BGC.")
    print("2. xxx.pfam.json ('xxx' is BGC id), which records Pfam domain component(s) of corresponding putative BGC.")
    print("3. xxx.attention_intra_pep_pfam_level.html ('xxx' is BGC id), which plots inter-domain attention flow(s) within single protein.")
    print("4. xxx.attention_multi_pep_pfam_level.html ('xxx' is BGC id), which plots inter-domain attention flow(s) within multiple proteins.")
    print("5. xxx.attention_pep_level.html ('xxx' is BGC id), which plots inter-protein attention flow(s).")
    print()

    l_BGC_id=[]
    dict_BGC_id2l_pep_id={}
    dict_BGC_id2putative_core_enzyme={}
    for contig in grouped_df_putative_BGC.groups:
        
        tdf_putative_BGC=grouped_df_putative_BGC.get_group(contig).reset_index(drop=True).copy()
        for i in range(len(tdf_putative_BGC)):
            BGC_id=tdf_putative_BGC["BGC_id"][i]
            start,end=tdf_putative_BGC["start"][i],tdf_putative_BGC["end"][i]
            record_seq=dict_record[contig]
            record_length=dict_record_length[contig]
            start_,end_=(start-flanking_DNA_sequence_length),(end+flanking_DNA_sequence_length)
            start_=start_ if start_>=1 else 1
            end_=end_ if end_<=record_length else record_length
            SeqIO.write([SeqRecord.SeqRecord(id=BGC_id,seq=record_seq[start_-1:end_],description='|'.join([contig,str(start_)+'-'+str(end_)]))],opath_prediction_result_BGC_detail+BGC_id+".dna.fasta","fasta")

            tl_pep_id=eval(tdf_putative_BGC["pep_id"][i])
            tmp_dict_pep2pfam_description={pep_id:dict_pep2pfam_description[pep_id] for pep_id in tl_pep_id}
            with open(opath_prediction_result_BGC_detail+BGC_id+".pfam.json",'w') as f:
                json.dump(tmp_dict_pep2pfam_description,f)
            tmp_dict_putative_core_enzyme=eval(tdf_putative_BGC["putative_core_enzyme"][i])
                
            l_BGC_id.append(BGC_id)
            dict_BGC_id2l_pep_id[BGC_id]=tl_pep_id
            dict_BGC_id2putative_core_enzyme[BGC_id]=tmp_dict_putative_core_enzyme

    a_intra_pep_pfam_level,a_multi_pep_pfam_level,a_pep_level=[],[],[]
    for no_submodel in range(n_submodel):
        path_model_=path_model+str(no_submodel)+"/genome_mining_model/"
        file_genome_mining_model=path_model_+"model.pkl"
        file_extended_appendix_json=path_model_+"extended_appendix.json"
        with open(file_extended_appendix_json,'r') as f:
            dict_extended_appendix=json.load(f)      
        genome_mining_model_.load_state_dict(th.load(file_genome_mining_model,map_location=device))
        genome_mining_model_.eval()

        genome_mining_model.corpus_pfam_id=genome_mining_model_.corpus_pfam_id
        genome_mining_model.linear_predefined_appendix_processing=genome_mining_model_.linear_predefined_appendix_processing
        genome_mining_model.linear_extended_appendix_processing=genome_mining_model_.linear_extended_appendix_processing
        genome_mining_model.l_transformer_encoder_layer_intra_pep_pfam_level=genome_mining_model_.l_transformer_encoder_layer_intra_pep_pfam_level
        genome_mining_model.l_transformer_encoder_layer_multi_pep_pfam_level=genome_mining_model_.l_transformer_encoder_layer_multi_pep_pfam_level
        genome_mining_model.l_transformer_encoder_layer_pfam2pep_level=genome_mining_model_.l_transformer_encoder_layer_pfam2pep_level
        genome_mining_model.l_transformer_encoder_layer_pep_level=genome_mining_model_.l_transformer_encoder_layer_pep_level  
        genome_mining_model.eval()

        for BGC_id in l_BGC_id:
            tl_pep_id=dict_BGC_id2l_pep_id[BGC_id]
            dict_data=model.genDataDict(
                l_pep_id=tl_pep_id,d_raw_predefined_appendix=d_raw_predefined_appendix,d_raw_extended_appendix=d_raw_extended_appendix,dict_pep2pfam=dict_pep2pfam,dict_esm=dict_esm,dict_extended_appendix=dict_extended_appendix,token_prefix="[PRE]",token_null="[NULL]"
            )
            tl_prefix_index=dict_data["prefix_index"]
            tl_pfam_index=dict_data["pfam_index"]
            tmp_dict_prefix_index2pfam_index_block=dict_data["dict_prefix_index2pfam_index_block"]
            tmp_dict_pfam_index2info=dict_data["dict_pfam_index2info"]
            tl_pep_id=dict_data["l_pep_id"]
            tl_token_pfam_id=dict_data["token_pfam_id"]
            
            with th.no_grad():
                ll_attn_weight_intra_pep_pfam_level,ll_attn_weight_multi_pep_pfam_level,ll_attn_weight_pfam2pep_level,ll_attn_weight_pep_level=genome_mining_model([dict_data],need_weight=True)[1]
            matrix_attn_weight_intra_pep_pfam_level=ll_attn_weight_intra_pep_pfam_level[0][0][0][0].cpu().detach().numpy()
            matrix_attn_weight_multi_pep_pfam_level=ll_attn_weight_multi_pep_pfam_level[0][0][0][0].cpu().detach().numpy()
            matrix_attn_weight_pep_level=ll_attn_weight_pep_level[0][0][0][0].cpu().detach().numpy()

            df_matrix_attn_weight_intra_pep_pfam_level=pd.DataFrame(matrix_attn_weight_intra_pep_pfam_level,index=tl_token_pfam_id,columns=tl_token_pfam_id)
            for pep_index,pep_id in enumerate(tl_pep_id):
                prefix_index=tl_prefix_index[pep_index]
                pfam_index_block=tmp_dict_prefix_index2pfam_index_block[prefix_index]
                for pfam_index1 in pfam_index_block:
                    pfam_id1=tl_token_pfam_id[pfam_index1]
                    start1,end1,relative_start1,relative_end1=tmp_dict_pfam_index2info[pfam_index1]
                    for pfam_index2 in pfam_index_block:
                        pfam_id2=tl_token_pfam_id[pfam_index2]
                        start2,end2,relative_start2,relative_end2=tmp_dict_pfam_index2info[pfam_index2]
                        attn_weight=df_matrix_attn_weight_intra_pep_pfam_level.iloc[pfam_index1,pfam_index2]
                        a_intra_pep_pfam_level.append([BGC_id,no_submodel,pep_index,pep_id,start1,end1,pfam_id1,start2,end2,pfam_id2,attn_weight])

            df_matrix_attn_weight_multi_pep_pfam_level=pd.DataFrame(matrix_attn_weight_multi_pep_pfam_level,index=tl_token_pfam_id,columns=tl_token_pfam_id)
            for pep_index1,pep_id1 in enumerate(tl_pep_id):
                prefix_index1=tl_prefix_index[pep_index1]
                pfam_index_block1=tmp_dict_prefix_index2pfam_index_block[prefix_index1]
                for pfam_index1 in pfam_index_block1:
                    pfam_id1=tl_token_pfam_id[pfam_index1]
                    start1,end1,relative_start1,relative_end1=tmp_dict_pfam_index2info[pfam_index1]
                    for pep_index2,pep_id2 in enumerate(tl_pep_id):
                        prefix_index2=tl_prefix_index[pep_index2]
                        pfam_index_block2=tmp_dict_prefix_index2pfam_index_block[prefix_index2]
                        for pfam_index2 in pfam_index_block2:
                            pfam_id2=tl_token_pfam_id[pfam_index2]
                            start2,end2,relative_start2,relative_end2=tmp_dict_pfam_index2info[pfam_index2]
                            attn_weight=df_matrix_attn_weight_multi_pep_pfam_level.iloc[pfam_index1,pfam_index2]
                            a_multi_pep_pfam_level.append([BGC_id,no_submodel,pep_index1,pep_id1,start1,end1,pfam_id1,pep_index2,pep_id2,start2,end2,pfam_id2,attn_weight])

            df_matrix_attn_weight_pep_level=pd.DataFrame(matrix_attn_weight_pep_level,index=tl_pep_id,columns=tl_pep_id)
            for pep_index1,pep_id1 in enumerate(tl_pep_id):
                for pep_index2,pep_id2 in enumerate(tl_pep_id):
                    attn_weight=df_matrix_attn_weight_pep_level.iloc[pep_index1,pep_index2]
                    a_pep_level.append([BGC_id,no_submodel,pep_index1,pep_id1,pep_index2,pep_id2,attn_weight])

    df_attn_weight_intra_pep_pfam_level=pd.DataFrame(
        a_intra_pep_pfam_level,columns=["BGC_id","no_submodel","pep_index","pep_id","pfam_start1","pfam_end1","pfam_id1","pfam_start2","pfam_end2","pfam_id2","attn_weight"]
    )
    df_attn_weight_multi_pep_pfam_level=pd.DataFrame(
        a_multi_pep_pfam_level,columns=["BGC_id","no_submodel","pep_index1","pep_id1","pfam_start1","pfam_end1","pfam_id1","pep_index2","pep_id2","pfam_start2","pfam_end2","pfam_id2","attn_weight"]
    )
    df_attn_weight_pep_level=pd.DataFrame(a_pep_level,columns=["BGC_id","no_submodel","pep_index1","pep_id1","pep_index2","pep_id2","attn_weight"])

    df_attn_weight_intra_pep_pfam_level=df_attn_weight_intra_pep_pfam_level.groupby(
        ["BGC_id","pep_index","pep_id","pfam_start1","pfam_end1","pfam_id1","pfam_start2","pfam_end2","pfam_id2"]
    ).mean().reset_index()[["BGC_id","pep_index","pep_id","pfam_start1","pfam_end1","pfam_id1","pfam_start2","pfam_end2","pfam_id2","attn_weight"]].sort_values(
        ["BGC_id","pep_index","pep_id","pfam_start1","pfam_end1","pfam_id1","pfam_start2","pfam_end2","pfam_id2"]
    ).reset_index(drop=True).copy()
    df_attn_weight_multi_pep_pfam_level=df_attn_weight_multi_pep_pfam_level.groupby(
        ["BGC_id","pep_index1","pep_id1","pfam_start1","pfam_end1","pfam_id1","pep_index2","pep_id2","pfam_start2","pfam_end2","pfam_id2"]
    ).mean().reset_index()[["BGC_id","pep_index1","pep_id1","pfam_start1","pfam_end1","pfam_id1","pep_index2","pep_id2","pfam_start2","pfam_end2","pfam_id2","attn_weight"]].sort_values(
        ["BGC_id","pep_index1","pep_id1","pfam_start1","pfam_end1","pfam_id1","pep_index2","pep_id2","pfam_start2","pfam_end2","pfam_id2"]
    ).reset_index(drop=True).copy()
    df_attn_weight_pep_level=df_attn_weight_pep_level.groupby(
        ["BGC_id","pep_index1","pep_id1","pep_index2","pep_id2"]
    ).mean().reset_index()[["BGC_id","pep_index1","pep_id1","pep_index2","pep_id2","attn_weight"]].sort_values(
        ["BGC_id","pep_index1","pep_id1","pep_index2","pep_id2"]
    ).reset_index(drop=True).copy()

    grouped_df_attn_weight_intra_pep_pfam_level=df_attn_weight_intra_pep_pfam_level.groupby("BGC_id")
    grouped_df_attn_weight_multi_pep_pfam_level=df_attn_weight_multi_pep_pfam_level.groupby("BGC_id")
    grouped_df_attn_weight_pep_level=df_attn_weight_pep_level.groupby("BGC_id")

    for BGC_id in l_BGC_id:
        tmp_dict_putative_core_enzyme=dict_BGC_id2putative_core_enzyme[BGC_id]
        tdf_attn_weight_intra_pep_pfam_level=grouped_df_attn_weight_intra_pep_pfam_level.get_group(BGC_id).reset_index(drop=True).copy()
        tdf_attn_weight_multi_pep_pfam_level=grouped_df_attn_weight_multi_pep_pfam_level.get_group(BGC_id).reset_index(drop=True).copy()
        tdf_attn_weight_pep_level=grouped_df_attn_weight_pep_level.get_group(BGC_id).reset_index(drop=True).copy()

        # = = = = = = = = = = = = = = = = = intra-pep pfam level = = = = = = = = = = = = = = = = =
        tdf_pfam_intra_pep_pfam_level=tdf_attn_weight_intra_pep_pfam_level[["pep_index","pep_id","pfam_start1","pfam_end1","pfam_id1"]].drop_duplicates().reset_index(drop=True).copy()
        tl_pep_id=[]
        tl_pfam_id=[]
        dict_pfam_info2index={}
        dict_pep_id2block_pfam_id={}
        for i in range(len(tdf_pfam_intra_pep_pfam_level)):
            pep_index,pfam_start,pfam_end=tdf_pfam_intra_pep_pfam_level["pep_index"][i],tdf_pfam_intra_pep_pfam_level["pfam_start1"][i],tdf_pfam_intra_pep_pfam_level["pfam_end1"][i]
            pep_id,pfam_id=tdf_pfam_intra_pep_pfam_level["pep_id"][i],tdf_pfam_intra_pep_pfam_level["pfam_id1"][i]
            tl_pfam_id.append(pfam_id)
            dict_pfam_info2index[(pep_index,pfam_start,pfam_end)]=i
            if pep_id not in dict_pep_id2block_pfam_id:
                tl_pep_id.append(pep_id)
                dict_pep_id2block_pfam_id[pep_id]=[]
            dict_pep_id2block_pfam_id[pep_id].append(pfam_id)
        tl_block_pfam_id=[dict_pep_id2block_pfam_id[pep_id] for pep_id in tl_pep_id]
        n_pep=len(tl_pep_id)
        n_pfam=len(tl_pfam_id)
        matrix_attn_weight=np.zeros([n_pfam,n_pfam])
        for i in range(len(tdf_attn_weight_intra_pep_pfam_level)):
            pep_index=tdf_attn_weight_intra_pep_pfam_level["pep_index"][i]
            pfam_start1,pfam_end1=tdf_attn_weight_intra_pep_pfam_level["pfam_start1"][i],tdf_attn_weight_intra_pep_pfam_level["pfam_end1"][i]
            pfam_start2,pfam_end2=tdf_attn_weight_intra_pep_pfam_level["pfam_start2"][i],tdf_attn_weight_intra_pep_pfam_level["pfam_end2"][i]
            attn_weight=tdf_attn_weight_intra_pep_pfam_level["attn_weight"][i]
            pfam_index1,pfam_index2=dict_pfam_info2index[(pep_index,pfam_start1,pfam_end1)],dict_pfam_info2index[(pep_index,pfam_start2,pfam_end2)]
            matrix_attn_weight[pfam_index1,pfam_index2]=attn_weight
        df_matrix_attn_weight_intra_pep_pfam_level=pd.DataFrame(matrix_attn_weight,index=tl_pfam_id,columns=tl_pfam_id)
        df_matrix_attn_weight_intra_pep_pfam_level.divide(df_matrix_attn_weight_intra_pep_pfam_level.sum(axis=1),axis=0)

        l_color=(l_base_color*int(np.ceil(n_pep/len(l_base_color))))[:n_pep]
        dict_pep2color=dict(zip(tl_pep_id,l_color))

        l_node_name=2*[dict_pfam_description[pfam_id] for pfam_id in tl_pfam_id]
        tl_node_category=[]
        for block_pfam_id,pep_id in zip(tl_block_pfam_id,tl_pep_id):
            tl_node_category+=len(block_pfam_id)*[pep_id]
        l_node_category=2*tl_node_category
        l_node_name_=[node_category+" ("+node_name+')' for node_category,node_name in zip(l_node_category,l_node_name)]

        index_level1=np.array(list(range(len(tl_pfam_id))))
        index_level2=index_level1+len(tl_pfam_id)
        df_matrix_attn_weight_intra_pep_pfam_level.index=index_level1
        df_matrix_attn_weight_intra_pep_pfam_level.columns=index_level2
        
        l_energy=[df_matrix_attn_weight_intra_pep_pfam_level.loc[idx].sum() for idx in df_matrix_attn_weight_intra_pep_pfam_level.index.values]+\
        [df_matrix_attn_weight_intra_pep_pfam_level[col].sum() for col in df_matrix_attn_weight_intra_pep_pfam_level.columns.values]
        dict_index2energy={i:l_energy[i] for i in range(len(l_energy))}

        thickness=20
        y_unit=1/n_pfam
        x=n_pfam*[0.05]+n_pfam*[0.95]
        y=[]
        for i in range(len(l_node_name)):
            if i%n_pfam==0:
                current_y_coord=1e-8
            offset=(dict_index2energy[i]*y_unit/2)
            current_y_coord+=offset
            y.append(current_y_coord)
            current_y_coord+=offset

        l_line_color=[]
        source,target,value=[],[],[]
        for source_index in df_matrix_attn_weight_intra_pep_pfam_level.index.values:
            for target_index in df_matrix_attn_weight_intra_pep_pfam_level.columns.values:
                source.append(source_index)
                target.append(target_index)
                value.append(df_matrix_attn_weight_intra_pep_pfam_level.loc[source_index,target_index])
                l_line_color.append(hex_to_rgba(dict_pep2color[l_node_category[source_index]],a=0.3))

        fig=go.Figure(
            data=[
                go.Sankey(
                    node={
                        "pad":0,
                        "thickness":thickness,
                        "label":l_node_name_,
                        'x':x,
                        'y':y,
                        "color":[hex_to_rgba(dict_pep2color[node_category],a=0.8) for node_category in l_node_category]
                    }, 
                    link={
                        "source":source,
                        "target":target,
                        "value":value,
                        "color":l_line_color
                    },
                    arrangement="fixed"
                )
            ]
        )
        fig.update_layout(
            autosize=True,
            width=None,
            height=None,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        config={'displayModeBar':False}
        fig.write_html(opath_prediction_result_BGC_detail+BGC_id+".attention_intra_pep_pfam_level.html",full_html=False,include_plotlyjs="cdn",config=config)
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        
        # = = = = = = = = = = = = = = = = = multi-pep pfam level = = = = = = = = = = = = = = = = =
        tdf_pfam_multi_pep_pfam_level=tdf_attn_weight_multi_pep_pfam_level[["pep_index1","pep_id1","pfam_start1","pfam_end1","pfam_id1"]].drop_duplicates().reset_index(drop=True).copy()
        tl_pep_id=[]
        tl_pfam_id=[]
        dict_pfam_info2index={}
        dict_pep_id2block_pfam_id={}
        for i in range(len(tdf_pfam_multi_pep_pfam_level)):
            pep_index,pfam_start,pfam_end=tdf_pfam_multi_pep_pfam_level["pep_index1"][i],tdf_pfam_multi_pep_pfam_level["pfam_start1"][i],tdf_pfam_multi_pep_pfam_level["pfam_end1"][i]
            pep_id,pfam_id=tdf_pfam_multi_pep_pfam_level["pep_id1"][i],tdf_pfam_multi_pep_pfam_level["pfam_id1"][i]
            tl_pfam_id.append(pfam_id)
            dict_pfam_info2index[(pep_index,pfam_start,pfam_end)]=i
            if pep_id not in dict_pep_id2block_pfam_id:
                tl_pep_id.append(pep_id)
                dict_pep_id2block_pfam_id[pep_id]=[]
            dict_pep_id2block_pfam_id[pep_id].append(pfam_id)
        tl_block_pfam_id=[dict_pep_id2block_pfam_id[pep_id] for pep_id in tl_pep_id]
        n_pep=len(tl_pep_id)
        n_pfam=len(tl_pfam_id)
        matrix_attn_weight=np.zeros([n_pfam,n_pfam])
        for i in range(len(tdf_attn_weight_multi_pep_pfam_level)):
            pep_index1=tdf_attn_weight_multi_pep_pfam_level["pep_index1"][i]
            pep_index2=tdf_attn_weight_multi_pep_pfam_level["pep_index2"][i]
            pfam_start1,pfam_end1=tdf_attn_weight_multi_pep_pfam_level["pfam_start1"][i],tdf_attn_weight_multi_pep_pfam_level["pfam_end1"][i]
            pfam_start2,pfam_end2=tdf_attn_weight_multi_pep_pfam_level["pfam_start2"][i],tdf_attn_weight_multi_pep_pfam_level["pfam_end2"][i]
            attn_weight=tdf_attn_weight_multi_pep_pfam_level["attn_weight"][i]
            pfam_index1,pfam_index2=dict_pfam_info2index[(pep_index1,pfam_start1,pfam_end1)],dict_pfam_info2index[(pep_index2,pfam_start2,pfam_end2)]
            matrix_attn_weight[pfam_index1,pfam_index2]=attn_weight
        df_matrix_attn_weight_multi_pep_pfam_level=pd.DataFrame(matrix_attn_weight,index=tl_pfam_id,columns=tl_pfam_id)
        df_matrix_attn_weight_multi_pep_pfam_level.divide(df_matrix_attn_weight_multi_pep_pfam_level.sum(axis=1),axis=0)

        l_color=(l_base_color*int(np.ceil(n_pep/len(l_base_color))))[:n_pep]
        dict_pep2color=dict(zip(tl_pep_id,l_color))
        
        l_node_name=2*[dict_pfam_description[pfam_id] for pfam_id in tl_pfam_id]
        tl_node_category=[]
        for block_pfam_id,pep_id in zip(tl_block_pfam_id,tl_pep_id):
            tl_node_category+=len(block_pfam_id)*[pep_id]
        l_node_category=2*tl_node_category
        l_node_name_=[node_category+" ("+node_name+')' for node_category,node_name in zip(l_node_category,l_node_name)]

        index_level1=np.array(list(range(len(tl_pfam_id))))
        index_level2=index_level1+len(tl_pfam_id)
        df_matrix_attn_weight_multi_pep_pfam_level.index=index_level1
        df_matrix_attn_weight_multi_pep_pfam_level.columns=index_level2

        l_energy=[df_matrix_attn_weight_multi_pep_pfam_level.loc[idx].sum() for idx in df_matrix_attn_weight_multi_pep_pfam_level.index.values]+\
        [df_matrix_attn_weight_multi_pep_pfam_level[col].sum() for col in df_matrix_attn_weight_multi_pep_pfam_level.columns.values]
        dict_index2energy={i:l_energy[i] for i in range(len(l_energy))}

        thickness=20
        y_unit=1/n_pfam
        x=n_pfam*[0.05]+n_pfam*[0.95]
        y=[]
        for i in range(len(l_node_name)):
            if i%n_pfam==0:
                current_y_coord=1e-8
            offset=(dict_index2energy[i]*y_unit/2)
            current_y_coord+=offset
            y.append(current_y_coord)
            current_y_coord+=offset

        l_line_color=[]
        source,target,value=[],[],[]
        for source_index in df_matrix_attn_weight_multi_pep_pfam_level.index.values:
            for target_index in df_matrix_attn_weight_multi_pep_pfam_level.columns.values:
                source.append(source_index)
                target.append(target_index)
                value.append(df_matrix_attn_weight_multi_pep_pfam_level.loc[source_index,target_index])
                l_line_color.append(hex_to_rgba(dict_pep2color[l_node_category[source_index]],a=0.3))

        fig=go.Figure(
            data=[
                go.Sankey(
                    node={
                        "pad":0,
                        "thickness":thickness,
                        "label":l_node_name_,
                        'x':x,
                        'y':y,
                        "color":[hex_to_rgba(dict_pep2color[node_category],a=0.8) for node_category in l_node_category]
                    }, 
                    link={
                        "source":source,
                        "target":target,
                        "value":value,
                        "color":l_line_color
                    },
                    arrangement="fixed"
                )
            ]
        )
        fig.update_layout(
            autosize=True,
            width=None,
            height=None,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        config={'displayModeBar':False}
        fig.write_html(opath_prediction_result_BGC_detail+BGC_id+".attention_multi_pep_pfam_level.html",full_html=False,include_plotlyjs="cdn",config=config)
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


        # = = = = = = = = = = = = = = = = = = = = pep level = = = = = = = = = = = = = = = = = = = =
        tdf_pep_pep_level=tdf_attn_weight_pep_level[["pep_index1","pep_id1"]].drop_duplicates().reset_index(drop=True).copy()
        tl_pep_id=[]
        dict_pep_info2index={}
        for i in range(len(tdf_pep_pep_level)):
            pep_index=tdf_pep_pep_level["pep_index1"][i]
            pep_id=tdf_pep_pep_level["pep_id1"][i]
            tl_pep_id.append(pep_id)
            dict_pep_info2index[pep_index]=i
        n_pep=len(tl_pep_id)
        matrix_attn_weight=np.zeros([n_pep,n_pep])
        for i in range(len(tdf_attn_weight_pep_level)):
            pep_index1=tdf_attn_weight_pep_level["pep_index1"][i]
            pep_index2=tdf_attn_weight_pep_level["pep_index2"][i]
            attn_weight=tdf_attn_weight_pep_level["attn_weight"][i]
            pep_index1_,pep_index2_=dict_pep_info2index[pep_index1],dict_pep_info2index[pep_index2]
            matrix_attn_weight[pep_index1_,pep_index2_]=attn_weight
        df_matrix_attn_weight_pep_level=pd.DataFrame(matrix_attn_weight,index=tl_pep_id,columns=tl_pep_id)
        df_matrix_attn_weight_pep_level.divide(df_matrix_attn_weight_pep_level.sum(axis=1),axis=0)

        l_color=(l_base_color*int(np.ceil(n_pep/len(l_base_color))))[:n_pep]
        dict_pep2color=dict(zip(tl_pep_id,l_color))

        l_node_name=2*tl_pep_id
        l_node_name_=2*[pep_id+" ("+','.join(tmp_dict_putative_core_enzyme[pep_id])+')' if pep_id in tmp_dict_putative_core_enzyme else pep_id for pep_id in tl_pep_id]
        index_level1=np.array(list(range(len(tl_pep_id))))
        index_level2=index_level1+len(tl_pep_id)
        df_matrix_attn_weight_pep_level.index=index_level1
        df_matrix_attn_weight_pep_level.columns=index_level2

        l_energy=[df_matrix_attn_weight_pep_level.loc[idx].sum() for idx in df_matrix_attn_weight_pep_level.index.values]+\
        [df_matrix_attn_weight_pep_level[col].sum() for col in df_matrix_attn_weight_pep_level.columns.values]
        dict_index2energy={i:l_energy[i] for i in range(len(l_energy))}

        thickness=20
        y_unit=1/n_pep
        x=n_pep*[0.05]+n_pep*[0.95]
        y=[]
        for i in range(len(l_node_name)):
            if i%n_pep==0:
                current_y_coord=1e-8
            offset=(dict_index2energy[i]*y_unit/2)
            current_y_coord+=offset
            y.append(current_y_coord)
            current_y_coord+=offset

        l_line_color=[]
        source,target,value=[],[],[]
        for source_index in df_matrix_attn_weight_pep_level.index.values:
            for target_index in df_matrix_attn_weight_pep_level.columns.values:
                source.append(source_index)
                target.append(target_index)
                value.append(df_matrix_attn_weight_pep_level.loc[source_index,target_index])
                l_line_color.append(hex_to_rgba(dict_pep2color[l_node_name[source_index]],a=0.3))

        fig=go.Figure(
            data=[
                go.Sankey(
                    node={
                        "pad":0,
                        "thickness":thickness,
                        "label":l_node_name_,
                        'x':x,
                        'y':y,
                        "color":[hex_to_rgba(dict_pep2color[node_name],a=0.8) for node_name in l_node_name]
                    }, 
                    link={
                        "source":source,
                        "target":target,
                        "value":value,
                        "color":l_line_color
                    },
                    arrangement="fixed"
                )
            ]
        )
        fig.update_layout(
            autosize=True,
            width=None,
            height=None,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        config={'displayModeBar':False}
        fig.write_html(opath_prediction_result_BGC_detail+BGC_id+".attention_pep_level.html",full_html=False,include_plotlyjs="cdn",config=config)
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    dict_pep2html_pfam_description={}
    for pep_id in dict_pep2pfam:
        n_char=0
        tl_pfam_id=[pfam_id for pfam_id,start,end,relative_start,relative_end in dict_pep2pfam[pep_id]]
        n_pfam=len(tl_pfam_id)
        if n_pfam==0:
            continue
        str_html_pfam_description=''
        for i,pfam_id in enumerate(tl_pfam_id):
            pfam_description=dict_pfam_description[pfam_id]
            if i!=n_pfam-1:
                tmp_str_html_pfam_description=pfam_description+', '
                str_html_pfam_description+=tmp_str_html_pfam_description
                n_char+=len(tmp_str_html_pfam_description)
            else:
                tmp_str_html_pfam_description=pfam_description
                str_html_pfam_description+=tmp_str_html_pfam_description
                n_char+=len(tmp_str_html_pfam_description)

            if i!=n_pfam-1:
                if n_char>=n_char_line_breaking_threshold:
                    str_html_pfam_description+="<br>"
                    n_char=0
        dict_pep2html_pfam_description[pep_id]=str_html_pfam_description
    
    print("The following html-based interactive images will be generated for the putative BGC-containing contig(s) and saved to {:s}:".format(opath_prediction_result_contig_detail))
    print("1. xxx.confidence_score.html ('xxx' is contig id), which plots ORF-level confidence scores given by f-BGM.")
    print("2. xxx.putative_BGC.html ('xxx' is contig id), which plots putative BGCs' genomic span predicted by f-BGM.")
    print()
    for contig in grouped_df_putative_BGC.groups:
        
        tdf_genome_mining_result=pd.read_csv(path_genome_mining_result+contig+".csv")
        l_pep_id=list(tdf_genome_mining_result["pep_id"].values)
        n_pep=len(l_pep_id)
        l_pep_idx=list(range(n_pep))
        dict_pep2idx=dict(zip(l_pep_id,l_pep_idx))
        l_pred_score=list(tdf_genome_mining_result["pred_score"].values)

        
        output_file(opath_prediction_result_contig_detail+contig+".confidence_score.html")
        p=figure(title=contig,x_axis_label="Peptide",y_axis_label="Confidence score",sizing_mode="stretch_both",tools=["xpan","box_zoom","xwheel_zoom","reset"])
        algorithm_name="f-BGM"
        source=ColumnDataSource(data={
            'x':l_pep_idx,
            'y':l_pred_score,
            "pep_id":l_pep_id,
        })
        p.line(source=source,x='x',y='y',line_width=1.5,legend_label=algorithm_name,line_color=dict_algorithm2color[algorithm_name])

        if flag_pred_score_threshold:
            source_pred_score_threshold=ColumnDataSource(data={
                'x':l_pep_idx,
                'y':[pred_score_threshold]*n_pep,
                "pep_id":["confidence score threshold"]*n_pep,
            })
            p.line(source=source_pred_score_threshold,x='x',y='y',line_width=1.5,line_color="red",line_dash="dotted")
        else:
            algorithm_name="f-BGM"
            source_pred_score_threshold=ColumnDataSource(data={
                'x':l_pep_idx,
                'y':[pred_score_threshold]*n_pep,
                "pep_id":[algorithm_name+" threshold"]*n_pep,
            })
            p.line(source=source_pred_score_threshold,x='x',y='y',line_width=1.5,line_color=dict_algorithm2color[algorithm_name],line_dash="dotted")

        hover_opts=dict(
            tooltips="@pep_id: @y{0.2f}",
        )
        hover=HoverTool(**hover_opts)
        p.add_tools(hover)
        p.xaxis.formatter = CustomJSTickFormatter(
            code='''
            var new_tick=tick
            '''+\
            "const l_pep_id="+str(l_pep_id)+\
            '''
            const l_pep_idx=[...Array(l_pep_id.length).keys()]
            if(tick in l_pep_idx)
                new_tick=l_pep_id[tick]
            else
                new_tick=''
            return new_tick
            '''
        )
        p.y_range=Range1d(-0.1,1.1)
        show(p)


        output_file(opath_prediction_result_contig_detail+contig+".putative_BGC.html")
        p=figure(title=contig,x_axis_label="Peptide",y_axis_label="Algorithm",sizing_mode="stretch_both",tools=["xpan","box_zoom","xwheel_zoom","reset"])
        algorithm_name="f-BGM"
        tdf_putative_BGC=grouped_df_putative_BGC.get_group(contig).reset_index(drop=True).copy()
        for i in range(len(tdf_putative_BGC)):
            BGC_id=tdf_putative_BGC["BGC_id"][i]
            tl_pep_id=eval(tdf_putative_BGC["pep_id"][i])
            tl_pep_idx=[dict_pep2idx[pep_id] for pep_id in tl_pep_id]
            min_pep_idx,max_pep_idx=np.min(tl_pep_idx),np.max(tl_pep_idx)
            tl_selected_pep_id=[pep_id for pep_id in tl_pep_id if pep_id in dict_pep2html_pfam_description]
            n_selected_pep=len(tl_selected_pep_id)
            l_color=(l_base_color*int(np.ceil(n_selected_pep/len(l_base_color))))[:n_selected_pep]
            dict_pep2color=dict(zip(tl_selected_pep_id,l_color))
            tmp_dict_putative_core_enzyme=dict_BGC_id2putative_core_enzyme[BGC_id]
            str_html_pfam_description="<div><i>"+BGC_id+"</i></div>"
            for pep_id in tl_selected_pep_id:
                str_html_pfam_description+=(
                    '<div style="color:'+dict_pep2color[pep_id]+';">'+(pep_id+" ("+','.join(tmp_dict_putative_core_enzyme[pep_id])+')' if pep_id in tmp_dict_putative_core_enzyme else pep_id)+":<br>"+dict_pep2html_pfam_description[pep_id]+"</div>"
                )
            source=ColumnDataSource(data={
                'x':[min_pep_idx-0.5,max_pep_idx+0.5],
                'y':[dict_algorithm2y[algorithm_name]]*2,
                "pfam_description":[str_html_pfam_description]*2,
            })
            p.line(source=source,x='x',y='y',line_width=5,line_color=dict_algorithm2color[algorithm_name])

        hover_opts=dict(
            tooltips="@pfam_description",
        )
        hover=HoverTool(**hover_opts)
        p.add_tools(hover)
        p.xaxis.formatter = CustomJSTickFormatter(
            code='''
            var new_tick=tick
            '''+\
            "const l_pep_id="+str(l_pep_id)+\
            '''
            const l_pep_idx=[...Array(l_pep_id.length).keys()]
            if(tick in l_pep_idx)
                new_tick=l_pep_id[tick]
            else
                new_tick=''
            return new_tick
            '''
        )
        p.yaxis.formatter = CustomJSTickFormatter(
            code='''
            var new_tick=tick
            '''+\
            "const l_algorithm_name="+str([""]+l_algorithm_name[::-1])+\
            '''
            const l_algorithm_idx=[...Array(l_algorithm_name.length).keys()]
            if(tick in l_algorithm_idx)
                new_tick=l_algorithm_name[tick]
            else
                new_tick=''
            return new_tick
            '''
        )
        p.x_range=Range1d(-0.5,n_pep-0.5)
        p.y_range=Range1d(0.5,n_algorithm+0.5)
        show(p)
    
    e_time=time.time()
    print("Finished, {:d}s taken.".format(int(e_time-s_time)))
    print()
