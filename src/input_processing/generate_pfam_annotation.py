import pandas as pd
import numpy as np
from Bio import SeqIO
import pyhmmer
from pyhmmer.hmmer import hmmscan
from pyhmmer.easel import SequenceFile
from pyhmmer.easel import Alphabet
from pyhmmer.plan7 import HMMFile
import json
import time

def genPfamJson(file_fasta,ofile_pfam_json,ref_file_pfam_A,n_thread,e_threshold):
    
    l_pep_id=[]
    dict_pep_seq={}
    for record in SeqIO.parse(file_fasta,"fasta"):
        record_id,record_seq=record.id,str(record.seq)
        l_pep_id.append(record_id)
        dict_pep_seq[record_id]=record_seq
    
    with SequenceFile(file_fasta,digital=True,alphabet=Alphabet.amino()) as seq_file,HMMFile(ref_file_pfam_A) as hmm_file:
        l_query_record=seq_file.read_block()
        target=hmm_file.optimized_profiles()
        a=[]
        for hits in hmmscan(l_query_record,target,E=e_threshold,cpus=n_thread):
            query_name=hits.query_name.decode()
            for hit in hits:
                accession,name,description=hit.accession.decode(),hit.name.decode(),hit.description.decode()
                evalue=hit.evalue
                if evalue>e_threshold:
                    continue
                for domain in hit.domains:
                    env_from,env_to=domain.env_from-1,domain.env_to
                    c_evalue,i_evalue=domain.c_evalue,domain.i_evalue
                    a.append([query_name,env_from,env_to,accession,evalue,c_evalue,i_evalue,name,description])
        df_hmmscan_result=pd.DataFrame(a,columns=["query_name","env_from","env_to","accession","evalue","c_evalue","i_evalue","name","description"])
        
    df_hmmscan_result=df_hmmscan_result[(df_hmmscan_result["evalue"]<=e_threshold)&(df_hmmscan_result["i_evalue"]<=e_threshold)].reset_index(drop=True).copy()
    grouped_df_hmmscan_result=df_hmmscan_result.groupby("query_name")
    dict_pep2pfam={}
    for pep_id in l_pep_id:
        if pep_id in grouped_df_hmmscan_result.groups:
            tdf_hmmscan_result=grouped_df_hmmscan_result.get_group(pep_id).sort_values(by=["env_from","env_to"],ascending=[True,True]).reset_index(drop=True).copy()
            pep_seq=dict_pep_seq[pep_id]
            pep_length=len(pep_seq)
            l_pfam=[]
            for i in range(len(tdf_hmmscan_result)):
                start,end,pfam_id=int(tdf_hmmscan_result["env_from"][i]),int(tdf_hmmscan_result["env_to"][i]),tdf_hmmscan_result["accession"][i]
                pfam_seq=pep_seq[start:end]
                relative_start,relative_end=start/pep_length,end/pep_length
                l_pfam.append((pfam_id,start,end,relative_start,relative_end,pfam_seq))         
            dict_pep2pfam[pep_id]=l_pfam
        else:
            dict_pep2pfam[pep_id]=[]
            
    with open(ofile_pfam_json,'w') as f:
        json.dump(dict_pep2pfam,f)
        
    return True
            
            
def genPfamAnnotation(file_contig_list,file_fasta_pep_seq,path_seq,ofile_pfam_json,ref_file_pfam_A,n_thread,e_threshold):
    
    print("Generating Pfam domain annotation for peptide sequence(s) by HMMER...")
    print()
    s_time=time.time()
    
    flag=genPfamJson(
        file_fasta=file_fasta_pep_seq,
        ofile_pfam_json=ofile_pfam_json,
        ref_file_pfam_A=ref_file_pfam_A,
        n_thread=n_thread,e_threshold=e_threshold,
    )
    if not flag:
        raise Exception("Failed to generate Pfam domain annotation for unknown reasons.")
    with open(ofile_pfam_json,'r') as f:
        dict_pep2pfam=json.load(f)
    
    l_contig=np.load(file_contig_list,allow_pickle=True)
    l_contig_with_pfam=[]
    l_exceptional_contig=[]
    for contig in l_contig:
        file_fasta_contig_pep_seq=path_seq+contig+".fasta"
        tl_pep_id=[record.id for record in SeqIO.parse(file_fasta_contig_pep_seq,"fasta")]
        flag_with_pfam=False
        for pep_id in tl_pep_id:
            if len(dict_pep2pfam[pep_id])>0:
                flag_with_pfam=True
                break
        if flag_with_pfam:
            l_contig_with_pfam.append(contig)
        else:
            l_exceptional_contig.append(contig)

    e_time=time.time()
    if len(l_contig_with_pfam)==0:
        print("No Pfam domain identified.")
        print()
    print("Finished, {:d}s taken.".format(int(e_time-s_time)))
    print()