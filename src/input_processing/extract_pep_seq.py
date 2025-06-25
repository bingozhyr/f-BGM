from Bio import SeqIO,Seq,SeqRecord
import pandas as pd
import numpy as np
import re
import random
import string
import os
import warnings
import time

def genRandomString(length):
    return ''.join([random.choice(string.ascii_letters) for i in range(length)])

n_omitting_threshold=20

def extractPepSeqFromFASTA(file_fasta,file_genome_annotation,file_type_genome_annotation,ofile_contig_fasta,ofile_df_pep_seq,ofile_contig_list,ofile_fasta_pep_seq,opath_seq):
    def getGFFDictAttribute(s):
        dict_attribute={}
        l_str_attribute=s.split(';')
        for str_attribute in l_str_attribute:
            if str_attribute=='':
                continue
            re_result=re.fullmatch('(.+?)=(.+?)',str_attribute)
            k,v=re_result.groups()
            k,v=k.strip(),v.strip()
            dict_attribute[k]=v
        return dict_attribute

    def getGTFDictAttribute(s):
        dict_attribute={}
        l_str_attribute=s.split(';')
        for str_attribute in l_str_attribute:
            if str_attribute=='':
                continue
            re_result=re.fullmatch('(.+?) \"(.+?)\"',str_attribute)
            k,v=re_result.groups()
            k,v=k.strip(),v.strip()
            dict_attribute[k]=v
        return dict_attribute

    def traverseDictFeature(dict_feature,feature_id,end_feature_type):
        l_route=[]
        def traverseDictFeature_(dict_feature,feature_id,end_feature_type,route):
            tmp_dict_child=dict_feature[feature_id]["dict_child"]
            for child_feature_type in tmp_dict_child:
                if child_feature_type!=end_feature_type:
                    for child_feature_id in tmp_dict_child[child_feature_type]:
                        route_=route.copy()
                        route_.append(child_feature_id)
                        traverseDictFeature_(dict_feature,child_feature_id,end_feature_type,route_)
                else:
                    route_=route.copy()
                    route_.append([child_feature_id for child_feature_id in tmp_dict_child[child_feature_type]])
                    l_route.append(route_)

        traverseDictFeature_(dict_feature,feature_id,end_feature_type,[feature_id])
        return l_route

    set_alphabet_residue={'A','B','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','X','Y','Z'}

    l_included_feature_type=["gene","transcript","mRNA","CDS"]
    dict_feature_type2id_field_gff={
        "gene":"ID",
        "transcript":"ID",
        "mRNA":"ID",
        "CDS":None,
    }
    dict_feature_type2id_field_gtf={
        "gene":"gene_id",
        "transcript":"transcript_id",
        "mRNA":"transcript_id",
        "CDS":None,
    }

    if file_type_genome_annotation=="gff":
        getDictAttribute=getGFFDictAttribute
        dict_feature_type2id_field=dict_feature_type2id_field_gff
    elif file_type_genome_annotation=="gtf":
        getDictAttribute=getGTFDictAttribute
        dict_feature_type2id_field=dict_feature_type2id_field_gtf

    print("Extracting peptide sequence(s) based on the fasta and "+file_type_genome_annotation+" files...")
    print()
    s_time=time.time()
    
    l_record=[record for record in SeqIO.parse(file_fasta,"fasta")]
    l_record_id=[record.id for record in l_record]
    dict_record_seq={record.id:record.seq for record in l_record}

    df_genome_annotation=pd.read_csv(file_genome_annotation,comment='#',sep='\t',header=None,dtype={0:str,1:str,2:str,3:int,4:int,5:str,6:str,7:str,8:str})
    df_genome_annotation=df_genome_annotation[df_genome_annotation[2].isin(l_included_feature_type)].reset_index(drop=True).copy()
    l_established_feature_id=[] 
    for i in range(len(df_genome_annotation)):
        contig,feature_type,start,end,strand=df_genome_annotation[0][i],df_genome_annotation[2][i],df_genome_annotation[3][i],df_genome_annotation[4][i],df_genome_annotation[6][i]
        dict_attribute=getDictAttribute(df_genome_annotation[8][i])
        id_field=dict_feature_type2id_field[feature_type]
        if id_field is not None:
            feature_id=dict_attribute[id_field]
            l_established_feature_id.append(feature_id)

    set_established_feature_id=set(l_established_feature_id)
    dict_feature={}
    for i in range(len(df_genome_annotation)):
        contig,feature_type,start,end,strand,phase=df_genome_annotation[0][i],df_genome_annotation[2][i],df_genome_annotation[3][i],df_genome_annotation[4][i],df_genome_annotation[6][i],df_genome_annotation[7][i]
        dict_attribute=getDictAttribute(df_genome_annotation[8][i])
        id_field=dict_feature_type2id_field[feature_type]
        if id_field is not None:
            feature_id=dict_attribute[id_field]
        else:
            while True:
                feature_id=feature_type+'-'+genRandomString(8)
                if feature_id in set_established_feature_id:
                    continue
                set_established_feature_id.add(feature_id)
                break
        dict_feature[feature_id]={"index":i,"feature_type":feature_type,"contig":contig,"start":start,"end":end,"strand":strand,"phase":phase,"dict_attribute":dict_attribute,"dict_child":{}}

    if file_type_genome_annotation=="gff":
        for feature_id in list(dict_feature.keys()):
            tmp_dict_feature=dict_feature[feature_id]
            dict_attribute=tmp_dict_feature["dict_attribute"]
            if "Parent" in dict_attribute:
                parent_feature_id=dict_attribute["Parent"]
                feature_type=tmp_dict_feature["feature_type"]
                if feature_type not in dict_feature[parent_feature_id]["dict_child"]:
                    dict_feature[parent_feature_id]["dict_child"][feature_type]=[]
                dict_feature[parent_feature_id]["dict_child"][feature_type].append(feature_id)
    elif file_type_genome_annotation=="gtf":
        for feature_id in list(dict_feature.keys()):
            tmp_dict_feature=dict_feature[feature_id]
            dict_attribute=tmp_dict_feature["dict_attribute"]
            if "transcript_id" in dict_attribute:
                gene_id=dict_attribute["gene_id"]
                transcript_id=dict_attribute["transcript_id"]
                feature_type=tmp_dict_feature["feature_type"]
                if transcript_id in dict_feature: # Transcript feature exists
                    if dict_feature_type2id_field[feature_type]=="transcript_id": # transcript feature itself
                        if feature_type not in dict_feature[gene_id]["dict_child"]:
                            dict_feature[gene_id]["dict_child"][feature_type]=[]
                        dict_feature[gene_id]["dict_child"][feature_type].append(feature_id)
                    else:
                        if feature_type not in dict_feature[transcript_id]["dict_child"]: # child of the transcript feature
                            dict_feature[transcript_id]["dict_child"][feature_type]=[]
                        dict_feature[transcript_id]["dict_child"][feature_type].append(feature_id)
                else: # Transcript feature not exists, try to fix it
                    tmp_dict_gene=dict_feature[gene_id]
                    dict_feature[transcript_id]={
                        "index":None,"feature_type":"auto_fixed_transcript","contig":tmp_dict_gene["contig"],"start":tmp_dict_gene["start"],"end":tmp_dict_gene["end"],"strand":tmp_dict_gene["strand"],"phase":tmp_dict_gene["phase"],"dict_attribute":{"gene_id":gene_id,"transcript_id":transcript_id},"dict_child":{}
                    }
                    if "auto_fixed_transcript" not in dict_feature[gene_id]["dict_child"]:
                        dict_feature[gene_id]["dict_child"]["auto_fixed_transcript"]=[]
                    dict_feature[gene_id]["dict_child"]["auto_fixed_transcript"].append(transcript_id)

                    dict_feature[transcript_id]["dict_child"][feature_type]=[]
                    dict_feature[transcript_id]["dict_child"][feature_type].append(feature_id)


    ol_fasta_record_contig=[SeqRecord.SeqRecord(id=record_id,seq=dict_record_seq[record_id],description='') for record_id in l_record_id]
                    
    a=[]
    l_exceptional_CDS_parent_feature_id=[]
    for feature_id in dict_feature:
        tmp_dict_feature=dict_feature[feature_id]
        feature_type,contig,start,end,strand,phase=tmp_dict_feature["feature_type"],tmp_dict_feature["contig"],tmp_dict_feature["start"],tmp_dict_feature["end"],tmp_dict_feature["strand"],tmp_dict_feature["phase"]
        record_seq=dict_record_seq[contig]
        if feature_type!="gene":
            continue
        l_gene2CDS_route=traverseDictFeature(dict_feature,feature_id,"CDS")
        for gene2CDS_route in l_gene2CDS_route:
            CDS_parent_feature_id=gene2CDS_route[-2]
            l_CDS_id=gene2CDS_route[-1]
            b=[]
            for CDS_id in l_CDS_id:
                tmp_dict_CDS=dict_feature[CDS_id]
                CDS_contig,CDS_start,CDS_end,CDS_strand,CDS_phase=tmp_dict_CDS["contig"],tmp_dict_CDS["start"]-1,tmp_dict_CDS["end"],tmp_dict_CDS["strand"],int(tmp_dict_CDS["phase"]) #transform to 0-based value for sequence extraction
                b.append([CDS_id,CDS_contig,CDS_start,CDS_end,CDS_strand,CDS_phase])
            tdf_CDS=pd.DataFrame(b,columns=["CDS_id","CDS_contig","CDS_start","CDS_end","CDS_strand","CDS_phase"]).sort_values(by="CDS_start",ascending=True).reset_index(drop=True).copy()
            if strand=='+':
                tdf_CDS=tdf_CDS.sort_values(by="CDS_start",ascending=True).reset_index(drop=True).copy()
            elif strand=='-':
                tdf_CDS=tdf_CDS.sort_values(by="CDS_start",ascending=False).reset_index(drop=True).copy()
            l_partial_CDS_start,l_partial_CDS_end=[],[]
            l_partial_CDS_seq=[]
            for i in range(len(tdf_CDS)):
                CDS_start,CDS_end,CDS_phase=tdf_CDS["CDS_start"][i],tdf_CDS["CDS_end"][i],tdf_CDS["CDS_phase"][i]
                if strand=='+':
                    partial_CDS_seq=record_seq[CDS_start:CDS_end]
                elif strand=='-':
                    partial_CDS_seq=record_seq[CDS_start:CDS_end].reverse_complement()
                if i==0:
                    if strand=='+':
                        CDS_start+=CDS_phase
                    elif strand=='-':
                        CDS_end-=CDS_phase
                    partial_CDS_seq=partial_CDS_seq[CDS_phase:]
                l_partial_CDS_start.append(CDS_start)
                l_partial_CDS_end.append(CDS_end)
                l_partial_CDS_seq.append(partial_CDS_seq)
            overall_CDS_start=(np.min(l_partial_CDS_start)+1) #transform to 1-based value for file writing
            overall_CDS_end=np.max(l_partial_CDS_end)
            CDS_seq=None
            for partial_CDS_seq in l_partial_CDS_seq:
                if CDS_seq is None:
                    CDS_seq=partial_CDS_seq
                else:
                    CDS_seq+=partial_CDS_seq
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pep_seq=CDS_seq.translate()
            CDS_seq=str(CDS_seq)
            pep_seq=str(pep_seq).rstrip('*')
            tmp_set_exceptional_residue=set(pep_seq)-set_alphabet_residue
            if len(tmp_set_exceptional_residue)>0:
                l_exceptional_CDS_parent_feature_id.append(CDS_parent_feature_id)
                source=''.join(list(tmp_set_exceptional_residue))
                target=len(source)*'X'
                pep_seq=pep_seq.translate(str.maketrans(source,target))
            a.append([feature_id,contig,start,end,strand,overall_CDS_start,overall_CDS_end,CDS_seq,pep_seq])
    n_exceptional_CDS_parent_feature_id=len(l_exceptional_CDS_parent_feature_id)
    if n_exceptional_CDS_parent_feature_id>0:
        if n_exceptional_CDS_parent_feature_id>n_omitting_threshold:
            str_l_exceptional_CDS_parent_feature_id=', '.join(l_exceptional_CDS_parent_feature_id[:n_omitting_threshold])+"..."
        else:
            str_l_exceptional_CDS_parent_feature_id=', '.join(l_exceptional_CDS_parent_feature_id)+'.'
        print("Identify {:d} feature(s) whose putative peptide sequence(s) containing exceptional residue(s): {:s}".format(n_exceptional_CDS_parent_feature_id,str_l_exceptional_CDS_parent_feature_id))
        print()
        print("Correct the exceptional residue(s) with 'X', which represents unknown amino acid.")
        print()
    df_pep_seq=pd.DataFrame(a,columns=["gene_id","contig","start","end","strand","CDS_start","CDS_end","CDS_seq","pep_seq"])
    df_pep_seq["contig"]=df_pep_seq["contig"].astype(pd.CategoricalDtype(categories=l_record_id,ordered=True))
    df_pep_seq=df_pep_seq.sort_values(by=["contig","start","end","strand","CDS_start","CDS_end"],ascending=[True,True,True,True,True,True]).reset_index(drop=True).copy()
    df_pep_seq["contig"]=df_pep_seq["contig"].astype(str)
    s_gene_count=df_pep_seq["gene_id"].value_counts()
    set_redundant_gene_id=set(s_gene_count[s_gene_count!=1].index.values)
    dict_gene_no_={gene_id:1 for gene_id in set_redundant_gene_id}
    l_pep_id=[]
    for gene_id in df_pep_seq["gene_id"].values:
        if gene_id not in dict_gene_no_:
            l_pep_id.append(gene_id)
        else:
            no_=dict_gene_no_[gene_id]
            l_pep_id.append(gene_id+'.'+str(no_))
            dict_gene_no_[gene_id]+=1
    df_pep_seq["pep_id"]=l_pep_id
    df_pep_seq=df_pep_seq[["gene_id","pep_id","contig","start","end","strand","CDS_start","CDS_end","CDS_seq","pep_seq"]].copy()
    
    l_contig=df_pep_seq["contig"].drop_duplicates().values
    
    e_time=time.time()
    n_pep_seq=len(df_pep_seq)
    if n_pep_seq>0:
        SeqIO.write(ol_fasta_record_contig,ofile_contig_fasta,"fasta")
        df_pep_seq.to_csv(ofile_df_pep_seq,index=None)
        np.save(ofile_contig_list,l_contig)
        grouped_df_pep_seq=df_pep_seq.groupby("contig")
        ol_fasta_record_pep_seq=[]
        for contig in grouped_df_pep_seq.groups:
            tdf_pep_seq=grouped_df_pep_seq.get_group(contig).reset_index(drop=True).copy()
            ol_fasta_record_contig_pep_seq=[]
            for i in range(len(tdf_pep_seq)):
                pep_id,pep_seq=tdf_pep_seq["pep_id"][i],tdf_pep_seq["pep_seq"][i]
                record_pep_seq=SeqRecord.SeqRecord(id=pep_id,seq=Seq.Seq(pep_seq),description='')
                ol_fasta_record_pep_seq.append(record_pep_seq)
                ol_fasta_record_contig_pep_seq.append(record_pep_seq)
            SeqIO.write(ol_fasta_record_contig_pep_seq,opath_seq+contig+".fasta","fasta")
        SeqIO.write(ol_fasta_record_pep_seq,ofile_fasta_pep_seq,"fasta")
        print("Finished, {:d}s taken. Extract {:d} peptide sequence(s) belonging to {:d} contig(s) in total.".format(int(e_time-s_time),n_pep_seq,len(l_contig)))
        print()
    else:
        raise Exception(
            "No peptide sequence identified. Please manully check the input file(s)."
        )

    
    
def extractPepSeqFromGenBank(file_genbank,ofile_contig_fasta,ofile_df_pep_seq,ofile_contig_list,ofile_fasta_pep_seq,opath_seq):
    def traverseDictFeature(dict_feature,feature_id,end_feature_type):
        l_route=[]
        def traverseDictFeature_(dict_feature,feature_id,end_feature_type,route):
            tmp_dict_child=dict_feature[feature_id]["dict_child"]
            for child_feature_type in tmp_dict_child:
                if child_feature_type!=end_feature_type:
                    for child_feature_id in tmp_dict_child[child_feature_type]:
                        route_=route.copy()
                        route_.append(child_feature_id)
                        traverseDictFeature_(dict_feature,child_feature_id,end_feature_type,route_)
                else:
                    for child_feature_id in tmp_dict_child[child_feature_type]:
                        route_=route.copy()
                        route_.append(child_feature_id)
                        l_route.append(route_)

        traverseDictFeature_(dict_feature,feature_id,end_feature_type,[feature_id])
        return l_route
    
    set_alphabet_residue={'A','B','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','X','Y','Z'}
    
    l_included_feature_type=["gene","CDS"]
    set_included_feature_type=set(l_included_feature_type)
    dict_feature_type2id_field={
        "gene":"locus_tag",
        "CDS":None,
    }

    print("Extracting peptide sequence(s) based on the genbank file...")
    print()
    s_time=time.time()

    l_record=list(SeqIO.parse(file_genbank,"genbank"))
    l_record_id=[record.id for record in l_record]
    dict_record_seq={record.id:record.seq for record in l_record}

    l_established_feature_id=[] 
    for record in l_record:
        record_id,record_seq=record.id,record.seq
        record_length=len(record_seq)
        for feature in record.features:
            feature_type=feature.type
            if feature_type in set_included_feature_type:
                location=feature.location
                dict_attribute=feature.qualifiers
                id_field=dict_feature_type2id_field[feature_type]
                if id_field is not None:
                    feature_id=dict_attribute[id_field][0]
                    l_established_feature_id.append(feature_id)

    set_established_feature_id=set(l_established_feature_id)
    dict_feature={}
    for record in l_record:
        record_id,record_seq=record.id,record.seq
        record_length=len(record_seq)
        for feature in record.features:
            feature_type=feature.type
            if feature_type in set_included_feature_type:
                location=feature.location
                dict_attribute=feature.qualifiers
                id_field=dict_feature_type2id_field[feature_type]
                if id_field is not None:
                    feature_id=dict_attribute[id_field][0]
                else:
                    while True:
                        feature_id=feature_type+'-'+genRandomString(8)
                        if feature_id in set_established_feature_id:
                            continue
                        set_established_feature_id.add(feature_id)
                        break
                dict_feature[feature_id]={"feature_type":feature_type,"contig":record_id,"location":location,"dict_attribute":dict_attribute,"dict_child":{}}

    for feature_id in list(dict_feature.keys()):
        tmp_dict_feature=dict_feature[feature_id]
        dict_attribute=tmp_dict_feature["dict_attribute"]
        if "locus_tag" in dict_attribute:
            feature_type=tmp_dict_feature["feature_type"]
            if dict_feature_type2id_field[feature_type]=="locus_tag": # gene feature itself
                pass
            else: # child of gene feature
                parent_feature_id=dict_attribute["locus_tag"][0]
                if feature_type not in dict_feature[parent_feature_id]["dict_child"]:
                    dict_feature[parent_feature_id]["dict_child"][feature_type]=[]
                dict_feature[parent_feature_id]["dict_child"][feature_type].append(feature_id)

                
    ol_fasta_record_contig=[SeqRecord.SeqRecord(id=record_id,seq=dict_record_seq[record_id],description='') for record_id in l_record_id]

    a=[]
    b=[]
    for feature_id in dict_feature:
        tmp_dict_feature=dict_feature[feature_id]
        feature_type,contig,location=tmp_dict_feature["feature_type"],tmp_dict_feature["contig"],tmp_dict_feature["location"]
        start,end,strand=location.start.real+1,location.end.real,location.strand #transform to 1-based value for file writing
        record_seq=dict_record_seq[contig]
        if feature_type!="gene":
            continue
        l_gene2CDS_route=traverseDictFeature(dict_feature,feature_id,"CDS")
        for gene2CDS_route in l_gene2CDS_route:
            CDS_id=gene2CDS_route[-1]
            tmp_dict_CDS=dict_feature[CDS_id]
            CDS_contig,CDS_location=tmp_dict_CDS["contig"],tmp_dict_CDS["location"]
            CDS_strand=CDS_location.strand
            dict_CDS_attribute=tmp_dict_CDS["dict_attribute"]
            CDS_codon_start=int(dict_CDS_attribute["codon_start"][0])-1 #transform to 0-based value for sequence extraction
            c=[]
            for i,location_part in enumerate(CDS_location.parts):
                CDS_start,CDS_end=location_part.start.real,location_part.end.real
                c.append([i,CDS_contig,CDS_start,CDS_end,CDS_strand])
            tdf_CDS=pd.DataFrame(c,columns=["CDS_part","CDS_contig","CDS_start","CDS_end","CDS_strand"]).sort_values(by="CDS_start",ascending=True).reset_index(drop=True).copy()
            if strand==1:
                tdf_CDS=tdf_CDS.sort_values(by="CDS_start",ascending=True).reset_index(drop=True).copy()
            elif strand==-1:
                tdf_CDS=tdf_CDS.sort_values(by="CDS_start",ascending=False).reset_index(drop=True).copy()
            l_partial_CDS_start,l_partial_CDS_end=[],[]
            l_partial_CDS_seq=[]
            for i in range(len(tdf_CDS)):
                CDS_start,CDS_end=tdf_CDS["CDS_start"][i],tdf_CDS["CDS_end"][i]
                if strand==1:
                    partial_CDS_seq=record_seq[CDS_start:CDS_end]
                elif strand==-1:
                    partial_CDS_seq=record_seq[CDS_start:CDS_end].reverse_complement()
                l_partial_CDS_start.append(CDS_start)
                l_partial_CDS_end.append(CDS_end)
                l_partial_CDS_seq.append(partial_CDS_seq)
            overall_CDS_start=np.min(l_partial_CDS_start)
            overall_CDS_end=np.max(l_partial_CDS_end)
            if strand==1:
                overall_CDS_start+=CDS_codon_start
            elif strand==-1:
                overall_CDS_end-=CDS_codon_start
            overall_CDS_start+=1 #transform to 1-based value for file writing
            CDS_seq=None
            for partial_CDS_seq in l_partial_CDS_seq:
                if CDS_seq is None:
                    CDS_seq=partial_CDS_seq
                else:
                    CDS_seq+=partial_CDS_seq
            CDS_seq=CDS_seq[CDS_codon_start:]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                putative_pep_seq=CDS_seq.translate()
            CDS_seq=str(CDS_seq)
            putative_pep_seq=str(putative_pep_seq).rstrip('*')
            tmp_set_exceptional_residue=set(putative_pep_seq)-set_alphabet_residue
            if len(tmp_set_exceptional_residue)>0:
                source=''.join(list(tmp_set_exceptional_residue))
                target=len(source)*'X'
                putative_pep_seq=putative_pep_seq.translate(str.maketrans(source,target))
            annotated_pep_seq=dict_CDS_attribute["translation"][0]
            if putative_pep_seq!=annotated_pep_seq:
                b.append([contig,'CDS',str(CDS_location),str(dict_CDS_attribute)])
            a.append([feature_id,contig,start,end,strand,overall_CDS_start,overall_CDS_end,CDS_seq,annotated_pep_seq])
    n_exceptional_pep_seq=len(b)
    if n_exceptional_pep_seq>0:
        df_exceptional_CDS=pd.DataFrame(b)
        print("Identify {:d} CDS feature(s) whose putative peptide sequence(s) inconsist with original annotation record(s):".format(n_exceptional_pep_seq))
        print()
        print(df_exceptional_CDS)
        print()
    df_pep_seq=pd.DataFrame(a,columns=["gene_id","contig","start","end","strand","CDS_start","CDS_end","CDS_seq","pep_seq"])
    df_pep_seq["contig"]=df_pep_seq["contig"].astype(pd.CategoricalDtype(categories=l_record_id,ordered=True))
    df_pep_seq=df_pep_seq.sort_values(by=["contig","start","end","strand","CDS_start","CDS_end"],ascending=[True,True,True,True,True,True]).reset_index(drop=True).copy()
    df_pep_seq["contig"]=df_pep_seq["contig"].astype(str)
    s_gene_count=df_pep_seq["gene_id"].value_counts()
    set_redundant_gene_id=set(s_gene_count[s_gene_count!=1].index.values)
    dict_gene_no_={gene_id:1 for gene_id in set_redundant_gene_id}
    l_pep_id=[]
    for gene_id in df_pep_seq["gene_id"].values:
        if gene_id not in dict_gene_no_:
            l_pep_id.append(gene_id)
        else:
            no_=dict_gene_no_[gene_id]
            l_pep_id.append(gene_id+'.'+str(no_))
            dict_gene_no_[gene_id]+=1
    df_pep_seq["pep_id"]=l_pep_id
    df_pep_seq=df_pep_seq[["gene_id","pep_id","contig","start","end","strand","CDS_start","CDS_end","CDS_seq","pep_seq"]].copy()
    
    l_contig=df_pep_seq["contig"].drop_duplicates().values
    
    e_time=time.time()
    n_pep_seq=len(df_pep_seq)
    if n_pep_seq>0:
        SeqIO.write(ol_fasta_record_contig,ofile_contig_fasta,"fasta")
        df_pep_seq.to_csv(ofile_df_pep_seq,index=None)
        np.save(ofile_contig_list,l_contig)
        grouped_df_pep_seq=df_pep_seq.groupby("contig")
        ol_fasta_record_pep_seq=[]
        for contig in grouped_df_pep_seq.groups:
            tdf_pep_seq=grouped_df_pep_seq.get_group(contig).reset_index(drop=True).copy()
            ol_fasta_record_contig_pep_seq=[]
            for i in range(len(tdf_pep_seq)):
                pep_id,pep_seq=tdf_pep_seq["pep_id"][i],tdf_pep_seq["pep_seq"][i]
                record_pep_seq=SeqRecord.SeqRecord(id=pep_id,seq=Seq.Seq(pep_seq),description='')
                ol_fasta_record_pep_seq.append(record_pep_seq)
                ol_fasta_record_contig_pep_seq.append(record_pep_seq)
            SeqIO.write(ol_fasta_record_contig_pep_seq,opath_seq+contig+".fasta","fasta")
        SeqIO.write(ol_fasta_record_pep_seq,ofile_fasta_pep_seq,"fasta")
        print("Finished, {:d}s taken. Extract {:d} peptide sequence(s) belonging to {:d} contig(s) in total.".format(int(e_time-s_time),n_pep_seq,len(l_contig)))
        print()
    else:
        raise Exception(
            "No peptide sequence identified. Please manully check the input file(s)."
        )