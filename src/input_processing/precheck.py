import Bio
from Bio import SeqIO
import pandas as pd
import numpy as np
import os
import re
import random
import string
import warnings
import time

def genRandomString(length):
    return ''.join([random.choice(string.ascii_letters) for i in range(length)])

n_omitting_threshold=20

dict_suffix2file_type={
    "gbk":"genbank",
    "gb":"genbank",
    "gbff":"genbank",
    "emb":"embl",
    "embl":"embl",
    "fa":"fasta",
    "fna":"fasta",
    "fasta":"fasta",
    "gtf":"gtf",
    "gff":"gff",
    "gff3":"gff",
}



def checkFASTAFile(file,sequence_type):
    
    set_alphabet_nucleotide={'A','C','G','T','a','c','g','t','N'}
    set_alphabet_pep={'A','B','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','X','Y','Z'}
    if sequence_type=="nucleotide":
        set_alphabet=set_alphabet_nucleotide
    elif sequence_type=="protein":
        set_alphabet=set_alphabet_pep
    
    print("Checking the fasta file...")
    print()
    s_time=time.time()
    try:
        l_record=list(SeqIO.parse(file,"fasta"))
    except:
        raise Exception(
            "Failed to read in. Please manually check the fasta file."
        )
    
    l_record_id=[]
    l_record_seq=[]
    for record in l_record:
        l_record_id.append(record.id)
        l_record_seq.append(record.seq)
    n_record=len(l_record_id)
    if n_record==0:
        raise Exception(
            "No sequence record found."
        )
    
    s_record_id=pd.Series(l_record_id).value_counts()
    l_duplicated_record_id=list(s_record_id[s_record_id>1].index.values)
    n_duplicated_record_id=len(l_duplicated_record_id)
    if n_duplicated_record_id>0:
        if n_duplicated_record_id>n_omitting_threshold:
            str_l_duplicated_record_id=', '.join(l_duplicated_record_id[:n_omitting_threshold])+"..."
        else:
            str_l_duplicated_record_id=', '.join(l_duplicated_record_id)+'.'
        raise Exception(
            "Identify {:d} duplicated sequence id(s): {:s}".format(n_duplicated_record_id,str_l_duplicated_record_id)
        )
    
    set_alphabet_record_seq=set()
    for record_seq in l_record_seq:
        set_alphabet_record_seq.update(set(record_seq))
    l_exceptional_char=list(set_alphabet_record_seq-set_alphabet)
    l_exceptional_char.sort()
    if len(l_exceptional_char)>0:
        if len(l_exceptional_char)>n_omitting_threshold:
            str_l_exceptional_char=', '.join(l_exceptional_char[:n_omitting_threshold])+"..."
        else:
            str_l_exceptional_char=', '.join(l_exceptional_char)+'.'
        raise Exception(
            "Identify exceptional char(s) in sequence(s): {:s}".format(str_l_exceptional_char)
        )
    
    e_time=time.time()
    print("Finished, {:d}s taken. Identify {:d} sequence record(s) in total.".format(int(e_time-s_time),n_record))
    print()
        
        

def checkGenomeAnnotationFile(file,file_type,ref_file_fasta):
    
    def checkGFFAttribute(s,set_required_attribute=None):
        dict_attribute={}
        l_str_attribute=s.split(';')
        for str_attribute in l_str_attribute:
            if str_attribute=='':
                continue
            re_result=re.fullmatch('(.+?)=(.+?)',str_attribute)
            if re_result is None:
                return False
            k,v=re_result.groups()
            k,v=k.strip(),v.strip()
            if k in dict_attribute:
                return False
            dict_attribute[k]=v
        if set_required_attribute is None:
            return True
        else:
            if set_required_attribute.issubset(set(dict_attribute.keys())):
                return True
            else:
                return False

    def checkGTFAttribute(s,set_required_attribute=None):
        dict_attribute={}
        l_str_attribute=s.split(';')
        for str_attribute in l_str_attribute:
            if str_attribute=='':
                continue
            re_result=re.fullmatch('(.+?) \"(.+?)\"',str_attribute)
            if re_result is None:
                return False
            k,v=re_result.groups()
            k,v=k.strip(),v.strip()
            if k in dict_attribute:
                return False
            dict_attribute[k]=v
        if set_required_attribute is None:
            return True
        else:
            if set_required_attribute.issubset(set(dict_attribute.keys())):
                return True
            else:
                return False

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

    l_included_feature_type=["gene","transcript","mRNA","CDS"]
    dict_feature_type2set_allowed_strand={
        "gene":{'+','-'},
        "transcript":{'+','-'},
        "mRNA":{'+','-'},
        "CDS":{'+','-'}
    }
    dict_feature_type2set_allowed_phase={
        "gene":{'.'},
        "transcript":{'.'},
        "mRNA":{'.'},
        "CDS":{'0','1','2'}
    }
    dict_feature_type2l_required_attribute_gff={
        "gene":["ID"],
        "transcript":["ID","Parent"],
        "mRNA":["ID","Parent"],
        "CDS":["Parent"],
    }
    dict_feature_type2set_required_attribute_gff={
        feature_type:set(dict_feature_type2l_required_attribute_gff[feature_type]) for feature_type in dict_feature_type2l_required_attribute_gff
    }
    dict_feature_type2l_required_attribute_gtf={
        "gene":["gene_id"],
        "transcript":["gene_id","transcript_id"],
        "mRNA":["gene_id","transcript_id"],
        "CDS":["gene_id","transcript_id"],
    }
    dict_feature_type2set_required_attribute_gtf={
        feature_type:set(dict_feature_type2l_required_attribute_gtf[feature_type]) for feature_type in dict_feature_type2l_required_attribute_gtf
    }
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
    l_feature_type_no_overlapping=["CDS"]

    if file_type=="gff":
        checkAttribute=checkGFFAttribute
        getDictAttribute=getGFFDictAttribute
        dict_feature_type2l_required_attribute=dict_feature_type2l_required_attribute_gff
        dict_feature_type2set_required_attribute=dict_feature_type2set_required_attribute_gff
        dict_feature_type2id_field=dict_feature_type2id_field_gff
    elif file_type=="gtf":
        checkAttribute=checkGTFAttribute
        getDictAttribute=getGTFDictAttribute
        dict_feature_type2l_required_attribute=dict_feature_type2l_required_attribute_gtf
        dict_feature_type2set_required_attribute=dict_feature_type2set_required_attribute_gtf
        dict_feature_type2id_field=dict_feature_type2id_field_gtf

    l_contig=[record.id for record in SeqIO.parse(ref_file_fasta,"fasta")]
    dict_contig_length={record.id:len(record.seq) for record in SeqIO.parse(ref_file_fasta,"fasta")}

    print("Checking the genome annotation file...")
    print()
    s_time=time.time()
    try:
        df_genome_annotation=pd.read_csv(file,comment='#',sep='\t',header=None,dtype={0:str,1:str,2:str,3:int,4:int,5:str,6:str,7:str,8:str})
        assert len(df_genome_annotation.columns)==9
    except:
        raise Exception(
            "Failed to read in. Please manually check the "+file_type+" file."
        )

    # Basic check: validity of entries
    index_exceptional_feature=~df_genome_annotation[0].isin(l_contig)
    n_exceptional_feature=np.sum(index_exceptional_feature)
    if n_exceptional_feature>0:
        raise Exception(
            "Identify {:d} feature(s) with unknown contig:".format(n_exceptional_feature)+"\n\n"+str(df_genome_annotation[index_exceptional_feature].reset_index(drop=True))
        )

    s_contig_length=df_genome_annotation[0].map(dict_contig_length).fillna(np.inf) # Fill the unknown contig(s) with length inf.
    index_exceptional_feature=(df_genome_annotation[4]<df_genome_annotation[3])|(df_genome_annotation[3]<1)|(df_genome_annotation[4]>s_contig_length)
    n_exceptional_feature=np.sum(index_exceptional_feature)
    if n_exceptional_feature>0:
        raise Exception(
            "Identify {:d} feature(s) with exceptional location:".format(n_exceptional_feature)+"\n\n"+str(df_genome_annotation[index_exceptional_feature].reset_index(drop=True))
        )

    index_exceptional_feature=~df_genome_annotation[6].isin({'+','-','.'})
    n_exceptional_feature=np.sum(index_exceptional_feature)
    if n_exceptional_feature>0:
        raise Exception(
            "Identify {:d} feature(s) with exceptional strand:".format(n_exceptional_feature)+"\n\n"+str(df_genome_annotation[index_exceptional_feature].reset_index(drop=True))
        )

    index_exceptional_feature=~df_genome_annotation[8].map(checkAttribute)
    n_exceptional_feature=np.sum(index_exceptional_feature)
    if n_exceptional_feature>0:
        raise Exception(
            "Identify {:d} feature(s) with exceptional attribute format:".format(n_exceptional_feature)+"\n\n"+str(df_genome_annotation[index_exceptional_feature].reset_index(drop=True))
        )
    
    
    grouped_df_genome_annotation=df_genome_annotation.groupby(2)
    # Advanced check: validity of strand
    flag_validity=True
    str_exception=""
    for feature_type in l_included_feature_type:
        if feature_type not in grouped_df_genome_annotation.groups:
            continue
        tdf_genome_annotation=grouped_df_genome_annotation.get_group(feature_type).reset_index(drop=True).copy()
        set_allowed_strand=dict_feature_type2set_allowed_strand[feature_type]
        index_exceptional_feature=[]
        for i in range(len(tdf_genome_annotation)):
            strand=tdf_genome_annotation[6][i]
            index_exceptional_feature.append(strand not in set_allowed_strand)
        n_exceptional_feature=np.sum(index_exceptional_feature) if len(index_exceptional_feature)!=0 else 0
        if n_exceptional_feature>0:
            if not flag_validity:
                str_exception+="\n\n\n"
            str_exception+="Identify {:d} {:s} feature(s) with exceptional strand:".format(n_exceptional_feature,feature_type)+"\n\n"
            str_exception+=str(tdf_genome_annotation[index_exceptional_feature].reset_index(drop=True))
            flag_validity=False
    if not flag_validity:
        raise Exception(str_exception)
        
    
    # Advanced check: validity of phase
    flag_validity=True
    str_exception=""
    for feature_type in l_included_feature_type:
        if feature_type not in grouped_df_genome_annotation.groups:
            continue
        tdf_genome_annotation=grouped_df_genome_annotation.get_group(feature_type).reset_index(drop=True).copy()
        set_allowed_phase=dict_feature_type2set_allowed_phase[feature_type]
        index_exceptional_feature=[]
        for i in range(len(tdf_genome_annotation)):
            phase=tdf_genome_annotation[7][i]
            if phase not in set_allowed_phase:
                index_exceptional_feature.append(True)
            elif phase!='.':
                start,end=tdf_genome_annotation[3][i]-1,tdf_genome_annotation[4][i] #transform to 0-based value for calculating feature length
                phase=int(phase)
                if (end-start)<phase:
                    index_exceptional_feature.append(True)
                else:
                    index_exceptional_feature.append(False)
            else:
                index_exceptional_feature.append(False)
        n_exceptional_feature=np.sum(index_exceptional_feature) if len(index_exceptional_feature)!=0 else 0
        if n_exceptional_feature>0:
            if not flag_validity:
                str_exception+="\n\n\n"
            str_exception+="Identify {:d} {:s} feature(s) with exceptional phase:".format(n_exceptional_feature,feature_type)+"\n\n"
            str_exception+=str(tdf_genome_annotation[index_exceptional_feature].reset_index(drop=True))
            flag_validity=False
    if not flag_validity:
        raise Exception(str_exception)


    # Advanced check: required attributes
    flag_validity=True
    str_exception=""
    for feature_type in l_included_feature_type:
        if feature_type not in grouped_df_genome_annotation.groups:
            continue
        tdf_genome_annotation=grouped_df_genome_annotation.get_group(feature_type).reset_index(drop=True).copy()
        l_required_attribute=dict_feature_type2l_required_attribute[feature_type]
        set_required_attribute=dict_feature_type2set_required_attribute[feature_type]
        index_exceptional_feature=[]
        for i in range(len(tdf_genome_annotation)):
            str_attribute=tdf_genome_annotation[8][i]
            index_exceptional_feature.append(not checkAttribute(str_attribute,set_required_attribute))
        n_exceptional_feature=np.sum(index_exceptional_feature) if len(index_exceptional_feature)!=0 else 0
        if n_exceptional_feature>0:
            if not flag_validity:
                str_exception+="\n\n\n"
            str_exception+="Identify {:d} {:s} feature(s) lacking required attribute(s)(i.e., {:s}):".format(n_exceptional_feature,feature_type,', '.join(l_required_attribute))+"\n\n"
            str_exception+=str(tdf_genome_annotation[index_exceptional_feature].reset_index(drop=True))
            flag_validity=False 
    if not flag_validity:
        raise Exception(str_exception)


    # Advanced check: duplicated/ambiguous ID
    df_genome_annotation=df_genome_annotation[df_genome_annotation[2].isin(l_included_feature_type)].reset_index(drop=True).copy()
    l_established_feature_id=[] 
    for i in range(len(df_genome_annotation)):
        contig,feature_type,start,end,strand=df_genome_annotation[0][i],df_genome_annotation[2][i],df_genome_annotation[3][i],df_genome_annotation[4][i],df_genome_annotation[6][i]
        dict_attribute=getDictAttribute(df_genome_annotation[8][i])
        id_field=dict_feature_type2id_field[feature_type]
        if id_field is not None:
            feature_id=dict_attribute[id_field]
            l_established_feature_id.append(feature_id)
    s_established_feature_id=pd.Series(l_established_feature_id).value_counts()
    l_duplicated_established_feature_id=s_established_feature_id[s_established_feature_id>1].index.values
    n_duplicated_established_feature_id=len(l_duplicated_established_feature_id)
    if n_duplicated_established_feature_id>0:
        if n_duplicated_established_feature_id>n_omitting_threshold:
            str_l_duplicated_established_feature_id=', '.join(l_duplicated_established_feature_id[:n_omitting_threshold])+"..."
        else:
            str_l_duplicated_established_feature_id=', '.join(l_duplicated_established_feature_id)+'.'
        raise Exception(
            "Identify {:d} duplicated feature ID(s): {:s}".format(n_duplicated_established_feature_id,str_l_duplicated_established_feature_id)
        )

    if file_type=="gtf":
        dict_transcript_id2gene_id={}
        for i in range(len(df_genome_annotation)):
            dict_attribute=getDictAttribute(df_genome_annotation[8][i])
            if "transcript_id" in dict_attribute:
                transcript_id=dict_attribute["transcript_id"]
                if "gene_id" not in dict_attribute:
                    gene_id=None
                else:
                    gene_id=dict_attribute["gene_id"]
                if transcript_id not in dict_transcript_id2gene_id:
                    dict_transcript_id2gene_id[transcript_id]=set()
                dict_transcript_id2gene_id[transcript_id].add(gene_id)
        l_exceptional_transcript_id=[transcript_id for transcript_id in dict_transcript_id2gene_id if len(dict_transcript_id2gene_id[transcript_id])!=1 or None in dict_transcript_id2gene_id[transcript_id]]
        n_exceptional_transcript_id=len(l_exceptional_transcript_id)
        if n_exceptional_transcript_id>0:
            if n_exceptional_transcript_id>n_omitting_threshold:
                str_l_exceptional_transcript_id=', '.join(l_exceptional_transcript_id[:n_omitting_threshold])+"..."
            else:
                str_l_exceptional_transcript_id=', '.join(l_exceptional_transcript_id)+'.'
            raise Exception(
                "Identify {:d} exceptional/ambiguous transcript ID(s): {:s}".format(n_exceptional_transcript_id,str_l_exceptional_transcript_id)
            )


    # Build dict_feature for further check
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


    # Advanced check: unknown parent
    if file_type=="gff":
        focused_field="Parent"
    elif file_type=="gtf":
        focused_field="gene_id"
    index_exceptional_feature=[]
    for i in range(len(df_genome_annotation)):
        flag_exceptional_feature=False
        dict_attribute=getDictAttribute(df_genome_annotation[8][i])
        if focused_field in dict_attribute:
            focused_feature_id=dict_attribute[focused_field]
            if focused_feature_id not in dict_feature:
                flag_exceptional_feature=True
        index_exceptional_feature.append(flag_exceptional_feature)
    n_exceptional_feature=np.sum(index_exceptional_feature) if len(index_exceptional_feature)!=0 else 0
    if n_exceptional_feature>0:
        raise Exception(
            "Identify {:d} feature(s) with unknown parent/ancestor:".format(n_exceptional_feature)+"\n\n"+str(df_genome_annotation[index_exceptional_feature].reset_index(drop=True))
        )


    # Build parent-child relationships for dict_feature
    if file_type=="gff":
        for feature_id in list(dict_feature.keys()):
            tmp_dict_feature=dict_feature[feature_id]
            dict_attribute=tmp_dict_feature["dict_attribute"]
            if "Parent" in dict_attribute:
                parent_feature_id=dict_attribute["Parent"]
                feature_type=tmp_dict_feature["feature_type"]
                if feature_type not in dict_feature[parent_feature_id]["dict_child"]:
                    dict_feature[parent_feature_id]["dict_child"][feature_type]=[]
                dict_feature[parent_feature_id]["dict_child"][feature_type].append(feature_id)
    elif file_type=="gtf":
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


    # Advanced check: parent-child consistency
    a=[]
    for feature_id in dict_feature:
        tmp_dict_feature=dict_feature[feature_id]
        dict_child=tmp_dict_feature["dict_child"]
        for child_feature_type in dict_child:
            for child_feature_id in dict_child[child_feature_type]:
                tmp_dict_child_feature=dict_feature[child_feature_id]
                flag_exceptional_parent_child_relationship=False
                if tmp_dict_child_feature["contig"]!=tmp_dict_feature["contig"]:
                    flag_exceptional_parent_child_relationship=True
                elif tmp_dict_child_feature["start"]<tmp_dict_feature["start"] or tmp_dict_child_feature["end"]>tmp_dict_feature["end"]:
                    flag_exceptional_parent_child_relationship=True
                elif tmp_dict_child_feature["strand"]!=tmp_dict_feature["strand"]:
                    flag_exceptional_parent_child_relationship=True
                if flag_exceptional_parent_child_relationship:
                    a.append([feature_id,child_feature_id,tmp_dict_child_feature["index"]])
    n_inconsistent_parent_child_pair=len(a)
    if n_inconsistent_parent_child_pair>0:
        df_inconsistent_parent_child_pair=pd.DataFrame(a,columns=["parent_feature_id","child_feature_id","index"])
        gdf_inconsistent_parent_child_pair=df_inconsistent_parent_child_pair.groupby("parent_feature_id")
        flag_omitting=False
        l_exceptional_parent_feature_id=list(gdf_inconsistent_parent_child_pair.groups.keys())
        n_exceptional_parent_feature_id=len(l_exceptional_parent_feature_id)
        if n_exceptional_parent_feature_id>n_omitting_threshold:
            flag_omitting=True
            l_exhibited_exceptional_parent_feature_id=l_exceptional_parent_feature_id[:n_omitting_threshold]
        else:
            l_exhibited_exceptional_parent_feature_id=l_exceptional_parent_feature_id
        str_exception=""
        str_exception+="Identify {:d} feature(s) with inconsistent children features:".format(n_exceptional_parent_feature_id)+"\n\n"
        for i,exceptional_parent_feature_id in enumerate(l_exhibited_exceptional_parent_feature_id):
            tdf_inconsistent_parent_child_pair=gdf_inconsistent_parent_child_pair.get_group(exceptional_parent_feature_id)
            index_exceptional_feature=np.sort(tdf_inconsistent_parent_child_pair["index"].values)
            str_exception+="Parent feature ID: "+exceptional_parent_feature_id+'\n'
            str_exception+="Inconsistent child feature(s):"+'\n'
            str_exception+=str(df_genome_annotation.iloc[index_exceptional_feature,:].reset_index(drop=True))
            if flag_omitting:
                str_exception+="\n\n\n"
            elif i!=len(l_exhibited_exceptional_parent_feature_id)-1:
                str_exception+="\n\n\n"
        if flag_omitting:
            str_exception+="......"
        raise Exception(str_exception)


    # Advanced check: overlapped location of child features
    a=[]
    for feature_id in dict_feature:
        tmp_dict_feature=dict_feature[feature_id]
        dict_child=tmp_dict_feature["dict_child"]
        for child_feature_type in l_feature_type_no_overlapping:
            if child_feature_type in dict_child:
                l_child_feature_id=dict_child[child_feature_type]
                set_pos=set()
                for child_feature_id in l_child_feature_id:
                    tmp_dict_child_feature=dict_feature[child_feature_id]
                    start,end=tmp_dict_child_feature["start"]-1,tmp_dict_child_feature["end"] #transform to 0-based value for calculating overlapped loaction
                    tmp_set_pos=set(range(start,end))
                    if len(tmp_set_pos&set_pos)>0:
                        a.append([feature_id,child_feature_type])
                        break
                    set_pos.update(tmp_set_pos)
    n_overlapped_location_exception=len(a)
    if n_overlapped_location_exception>0:
        df_overlapped_location_exception=pd.DataFrame(a,columns=["parent_feature_id","child_feature_type"])
        gdf_overlapped_location_exception=df_overlapped_location_exception.groupby("child_feature_type")
        str_exception=""
        for child_feature_type in gdf_overlapped_location_exception.groups:
            tdf_overlapped_location_exception=gdf_overlapped_location_exception.get_group(child_feature_type)
            l_exceptional_parent_feature_id=tdf_overlapped_location_exception["parent_feature_id"].values
            n_exceptional_parent_feature_id=len(l_exceptional_parent_feature_id)
            if n_exceptional_parent_feature_id>n_omitting_threshold:
                str_l_exceptional_parent_feature_id=', '.join(l_exceptional_parent_feature_id[:n_omitting_threshold])+"..."
            else:
                str_l_exceptional_parent_feature_id=', '.join(l_exceptional_parent_feature_id)+'.'
            if str_exception!="":
                str_exception+='\n'
            str_exception+="Identify {:d} feature(s) whose children {:s} features are overlapped each other: {:s}".format(n_exceptional_parent_feature_id,child_feature_type,str_l_exceptional_parent_feature_id)+'\n'
        raise Exception(str_exception)
    
    
    # Advanced check: phase validity of first CDS part & CDS part length%3==0
    l_exceptional_feature_id1,l_exceptional_feature_id2=[],[]
    for feature_id in dict_feature:
        tmp_dict_feature=dict_feature[feature_id]
        contig,start,end,strand,phase=tmp_dict_feature["contig"],tmp_dict_feature["start"],tmp_dict_feature["end"],tmp_dict_feature["strand"],tmp_dict_feature["phase"]
        dict_child=tmp_dict_feature["dict_child"]
        if "CDS" in dict_child:
            l_child_feature_id=dict_child["CDS"]
            a=[]
            for child_feature_id in l_child_feature_id:
                tmp_dict_child_feature=dict_feature[child_feature_id]
                CDS_contig,CDS_start,CDS_end,CDS_strand,CDS_phase=tmp_dict_child_feature["contig"],tmp_dict_child_feature["start"]-1,tmp_dict_child_feature["end"],tmp_dict_child_feature["strand"],int(tmp_dict_child_feature["phase"]) #transform to 0-based value for calculating the length of CDS part
                a.append([CDS_contig,CDS_start,CDS_end,CDS_strand,CDS_phase])
            tdf_CDS=pd.DataFrame(a,columns=["CDS_contig","CDS_start","CDS_end","CDS_strand","CDS_phase"])
            if strand=='+':
                tdf_CDS=tdf_CDS.sort_values(by="CDS_start",ascending=True).reset_index(drop=True).copy()
            elif strand=='-':
                tdf_CDS=tdf_CDS.sort_values(by="CDS_start",ascending=False).reset_index(drop=True).copy()
            first_CDS_start,first_CDS_end,first_CDS_phase=tdf_CDS["CDS_start"][0],tdf_CDS["CDS_end"][0],tdf_CDS["CDS_phase"][0]
            if (first_CDS_end-first_CDS_start)<(first_CDS_phase+1):
                l_exceptional_feature_id1.append(feature_id)
            n_CDS=len(tdf_CDS)
            for i in range(len(tdf_CDS)):
                CDS_start,CDS_end,CDS_phase=tdf_CDS["CDS_start"][i],tdf_CDS["CDS_end"][i],tdf_CDS["CDS_phase"][i]
                if i!=n_CDS-1:
                    next_CDS_phase=tdf_CDS["CDS_phase"][i+1]
                    partial_CDS_length=CDS_end-CDS_start-CDS_phase+next_CDS_phase
                    if partial_CDS_length%3!=0:
                        l_exceptional_feature_id2.append(feature_id)
                        break
                else:
                    partial_CDS_length=CDS_end-CDS_start-CDS_phase
    n_exceptional_feature_id1=len(l_exceptional_feature_id1)
    if n_exceptional_feature_id1>0:
        if n_exceptional_feature_id1>n_omitting_threshold:
            str_l_exceptional_feature_id1=', '.join(l_exceptional_feature_id1[:n_omitting_threshold])+"..."
        else:
            str_l_exceptional_feature_id1=', '.join(l_exceptional_feature_id1)+'.'
        raise Exception(
            "Identify {:d} feature(s) whose first CDS part with exceptional phase: {:s}".format(n_exceptional_feature_id1,str_l_exceptional_feature_id1)
        )   
    n_exceptional_feature_id2=len(l_exceptional_feature_id2)
    if n_exceptional_feature_id2>0:
        if n_exceptional_feature_id2>n_omitting_threshold:
            str_l_exceptional_feature_id2=', '.join(l_exceptional_feature_id2[:n_omitting_threshold])+"..."
        else:
            str_l_exceptional_feature_id2=', '.join(l_exceptional_feature_id2)+'.'
        raise Exception(
            "Identify {:d} feature(s) containing at least one CDS part whose length is not multiple of 3: {:s}".format(n_exceptional_feature_id2,str_l_exceptional_feature_id2)
        )
    
    e_time=time.time()
    print("Finished, {:d}s taken.".format(int(e_time-s_time)))
    print()



def checkGenBankFile(file):

    def checkFeatureLocation(object_location,record_length,required_n_location_part=None):

        if type(object_location)==Bio.SeqFeature.SimpleLocation:
            l_location_part=[object_location]
        elif type(object_location)==Bio.SeqFeature.CompoundLocation:
            l_location_part=feature.location.parts
        else:
            return False

        n_location_part=len(l_location_part)
        if required_n_location_part is not None:
            if n_location_part!=required_n_location_part:
                return False

        set_strand=set()
        for location_part in l_location_part:
            set_strand.add(location_part.strand)
        if len(set_strand)!=1 or not set_strand.issubset({-1,1}):
            return False

        set_pos=set()
        for location_part in l_location_part:
            start,end=location_part.start.real,location_part.end.real
            if start>=end:
                return False
            if start<0 or end>record_length:
                return False
            tmp_set_pos=set(location_part)
            if len(set_pos&tmp_set_pos)>0:
                return False
            set_pos.update(tmp_set_pos)
        return True
    
    set_alphabet_nucleotide={'A','C','G','T','a','c','g','t','N'}
    set_alphabet_pep={'A','B','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','X','Y','Z'}

    l_included_feature_type=["gene","CDS"]
    set_included_feature_type=set(l_included_feature_type)
    dict_feature_type2n_required_location_part={
        "gene":1,
        "CDS":None,
    }
    dict_feature_type2l_required_attribute={
        "gene":["locus_tag"],
        "CDS":["locus_tag","codon_start","translation"],
    }
    dict_feature_type2set_required_attribute={
        feature_type:set(dict_feature_type2l_required_attribute[feature_type]) for feature_type in dict_feature_type2l_required_attribute
    }
    dict_feature_type2id_field={
        "gene":"locus_tag",
        "CDS":None,
    }

    print("Checking the genbank file...")
    print()
    s_time=time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            l_record=list(SeqIO.parse(file,"genbank"))
    except:
        raise Exception(
            "Failed to read in. Please manually check the genbank file."
        )

    l_record_id=[]
    l_record_seq=[]
    for record in l_record:
        l_record_id.append(record.id)
        l_record_seq.append(record.seq)
    n_record=len(l_record_id)
    if n_record==0:
        raise Exception(
            "No sequence record found."
        )

    s_record_id=pd.Series(l_record_id).value_counts()
    l_duplicated_record_id=list(s_record_id[s_record_id>1].index.values)
    n_duplicated_record_id=len(l_duplicated_record_id)
    if n_duplicated_record_id>0:
        if n_duplicated_record_id>n_omitting_threshold:
            str_l_duplicated_record_id=', '.join(l_duplicated_record_id[:n_omitting_threshold])+"..."
        else:
            str_l_duplicated_record_id=', '.join(l_duplicated_record_id)+'.'
        raise Exception(
            "Identify {:d} duplicated sequence id(s): {:s}".format(n_duplicated_record_id,str_l_duplicated_record_id)
        )

    set_alphabet_record_seq=set()
    for record_seq in l_record_seq:
        set_alphabet_record_seq.update(set(record_seq))
    l_exceptional_char=list(set_alphabet_record_seq-set_alphabet_nucleotide)
    l_exceptional_char.sort()
    if len(l_exceptional_char)>0:
        if len(l_exceptional_char)>n_omitting_threshold:
            str_l_exceptional_char=', '.join(l_exceptional_char[:n_omitting_threshold])+"..."
        else:
            str_l_exceptional_char=', '.join(l_exceptional_char)+'.'
        raise Exception(
            "Identify exceptional char(s) in sequence(s): {:s}".format(str_l_exceptional_char)
        )


    # Basic check: validity of feature location
    a=[]
    for record in l_record:
        record_id,record_seq=record.id,record.seq
        record_length=len(record_seq)
        for feature in record.features:
            feature_type=feature.type
            location=feature.location
            dict_attribute=feature.qualifiers 
            flag_exceptional_feature=not checkFeatureLocation(location,record_length=record_length,required_n_location_part=dict_feature_type2n_required_location_part[feature_type] if feature_type in set_included_feature_type else None)
            if flag_exceptional_feature:
                a.append([record_id,feature_type,str(location),str(dict_attribute)])
    n_exceptional_feature=len(a)
    if n_exceptional_feature>0:
        df_exceptional_feature=pd.DataFrame(a)
        raise Exception(
            "Identify {:d} feature(s) with exceptional location:".format(n_exceptional_feature)+"\n\n"+str(df_exceptional_feature)
        )


    # Advanced check: required attribute
    flag_validity=True
    str_exception=""
    for focused_feature_type in l_included_feature_type:
        l_required_attribute=dict_feature_type2l_required_attribute[focused_feature_type]
        set_required_attribute=dict_feature_type2set_required_attribute[focused_feature_type]
        a=[]
        for record in l_record:
            record_id,record_seq=record.id,record.seq
            record_length=len(record_seq)
            for feature in record.features:
                feature_type=feature.type
                if feature_type==focused_feature_type:
                    location=feature.location
                    dict_attribute=feature.qualifiers
                    if not set_required_attribute.issubset(set(dict_attribute.keys())):
                        a.append([record_id,feature_type,str(location),str(dict_attribute)])
        n_exceptional_feature=len(a)
        if n_exceptional_feature>0:
            if not flag_validity:
                str_exception+="\n\n\n"
            str_exception+="Identify {:d} {:s} feature(s) lacking required attribute(s)(i.e., {:s}):".format(n_exceptional_feature,focused_feature_type,', '.join(l_required_attribute))+"\n\n"
            str_exception+=str(pd.DataFrame(a))
            flag_validity=False         
    if not flag_validity:
        raise Exception(str_exception)
    

    # Advanced check: required attribute with wrong number of value(s)
    a=[]
    for record in l_record:
        record_id,record_seq=record.id,record.seq
        record_length=len(record_seq)
        for feature in record.features:
            feature_type=feature.type
            if feature_type in set_included_feature_type:
                location=feature.location
                dict_attribute=feature.qualifiers
                for attribute in dict_feature_type2l_required_attribute[feature_type]:
                    l_value=dict_attribute[attribute]
                    if len(l_value)!=1:
                        a.append([record_id,feature_type,str(location),str(dict_attribute)])
    n_exceptional_feature=len(a)
    if n_exceptional_feature>0:
        raise Exception(
            "Identify {:d} feature(s) whose required attribute(s) containing wrong number of value(s) (single value expected):".format(n_exceptional_feature)+"\n\n"+str(pd.DataFrame(a))
        )
    
    
    # Advanced check: validity of codon_start
    flag_validity=True
    str_exception=""
    for focused_feature_type in l_included_feature_type:
        set_required_attribute=dict_feature_type2set_required_attribute[focused_feature_type]
        if "codon_start" in set_required_attribute:
            a=[]
            for record in l_record:
                record_id,record_seq=record.id,record.seq
                record_length=len(record_seq)
                for feature in record.features:
                    feature_type=feature.type
                    if feature_type==focused_feature_type:
                        location=feature.location
                        dict_attribute=feature.qualifiers
                        try:
                            codon_start=int(dict_attribute["codon_start"][0])
                            assert codon_start>0
                            strand=location.strand
                            b=[]
                            for location_part in location.parts:
                                part_start,part_end=location_part.start.real,location_part.end.real
                                b.append([part_start,part_end])
                            df_location_part=pd.DataFrame(b,columns=["start","end"])
                            if strand==1:
                                df_location_part=df_location_part.sort_values(by="start",ascending=True).reset_index(drop=True).copy()
                            elif strand==-1:
                                df_location_part=df_location_part.sort_values(by="start",ascending=False).reset_index(drop=True).copy()
                            first_part_start,first_part_end=df_location_part["start"][0],df_location_part["end"][0]
                            assert (first_part_end-first_part_start)>=codon_start
                        except:
                            a.append([record_id,feature_type,str(location),str(dict_attribute)])
            n_exceptional_feature=len(a)
            if n_exceptional_feature>0:
                if not flag_validity:
                    str_exception+="\n\n\n"
                str_exception+="Identify {:d} {:s} feature(s) with exceptional 'codon_start':".format(n_exceptional_feature,focused_feature_type)+"\n\n"
                str_exception+=str(pd.DataFrame(a))
                flag_validity=False
    if not flag_validity:
        raise Exception(str_exception)
    
    
    # # Advanced check: CDS length%3==0
    # a=[]
    # for record in l_record:
    #     record_id,record_seq=record.id,record.seq
    #     record_length=len(record_seq)
    #     for feature in record.features:
    #         feature_type=feature.type
    #         if feature_type=="CDS":
    #             location=feature.location
    #             dict_attribute=feature.qualifiers
    #             codon_start=int(dict_attribute["codon_start"][0])-1 #transform to 0-based value for calculating CDS length
    #             CDS_length=len(location)-codon_start
    #             if CDS_length%3!=0:
    #                 a.append([record_id,feature_type,str(location),str(dict_attribute)])
    # n_exceptional_feature=len(a)
    # if n_exceptional_feature>0:
    #     raise Exception(
    #         "Identify {:d} CDS feature(s) whose length is not multiple of 3:".format(n_exceptional_feature)+"\n\n"+str(pd.DataFrame(a))
    #     )
    

    # Advanced check: validity of translation
    flag_validity=True
    str_exception=""
    for focused_feature_type in l_included_feature_type:
        set_required_attribute=dict_feature_type2set_required_attribute[focused_feature_type]
        if "translation" in set_required_attribute:
            set_exceptional_char=set()
            a=[]
            for record in l_record:
                record_id,record_seq=record.id,record.seq
                record_length=len(record_seq)
                for feature in record.features:
                    feature_type=feature.type
                    if feature_type==focused_feature_type:
                        location=feature.location
                        dict_attribute=feature.qualifiers
                        tmp_set_exceptional_char=set(dict_attribute["translation"][0])-set_alphabet_pep
                        set_exceptional_char.update(tmp_set_exceptional_char)
                        if len(tmp_set_exceptional_char)>0:
                            a.append([record_id,feature_type,str(location),str(dict_attribute)])
            n_exceptional_feature=len(a)
            if n_exceptional_feature>0:
                l_exceptional_char=list(set_exceptional_char)
                if len(l_exceptional_char)>n_omitting_threshold:
                    str_l_exceptional_char=', '.join(l_exceptional_char[:n_omitting_threshold])+"..."
                else:
                    str_l_exceptional_char=', '.join(l_exceptional_char)+'.'
                if not flag_validity:
                    str_exception+="\n\n\n"
                str_exception+="Identify {:d} {:s} feature(s) whose 'translation' containing exceptional char(s): {:s}".format(n_exceptional_feature,focused_feature_type,str_l_exceptional_char)+"\n\n"
                str_exception+=str(pd.DataFrame(a))
                flag_validity=False
    if not flag_validity:
        raise Exception(str_exception)


    # Advanced check: duplicated ID
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
    s_established_feature_id=pd.Series(l_established_feature_id).value_counts()
    l_duplicated_established_feature_id=s_established_feature_id[s_established_feature_id>1].index.values
    n_duplicated_established_feature_id=len(l_duplicated_established_feature_id)
    if n_duplicated_established_feature_id>0:
        if n_duplicated_established_feature_id>n_omitting_threshold:
            str_l_duplicated_established_feature_id=', '.join(l_duplicated_established_feature_id[:n_omitting_threshold])+"..."
        else:
            str_l_duplicated_established_feature_id=', '.join(l_duplicated_established_feature_id)+'.'
        raise Exception("Identify {:d} duplicated feature ID(s): {:s}".format(n_duplicated_established_feature_id,str_l_duplicated_established_feature_id))


    # Build dict_feature for further check
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


    # Advanced check: unknown parent
    a=[]
    for record in l_record:
        record_id,record_seq=record.id,record.seq
        record_length=len(record_seq)
        for feature in record.features:
            feature_type=feature.type
            if feature_type in set_included_feature_type:
                location=feature.location
                dict_attribute=feature.qualifiers
                if "locus_tag" in dict_attribute:
                    focused_feature_id=dict_attribute["locus_tag"][0]
                    if focused_feature_id not in dict_feature:
                        a.append([record_id,feature_type,str(location),str(dict_attribute)])
    n_exceptional_feature=len(a)
    if n_exceptional_feature>0:
        raise Exception(
            "Identify {:d} feature(s) with unknown parent/ancestor:".format(n_exceptional_feature)+"\n\n"+str(pd.DataFrame(a))
        )


    # Build parent-child relationships for dict_feature
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


    # Advanced check: parent-child consistency
    a=[]
    for feature_id in dict_feature:
        tmp_dict_feature=dict_feature[feature_id]
        dict_child=tmp_dict_feature["dict_child"]
        for child_feature_type in dict_child:
            for child_feature_id in dict_child[child_feature_type]:
                tmp_dict_child_feature=dict_feature[child_feature_id]
                location,child_location=tmp_dict_feature["location"],tmp_dict_child_feature["location"]
                flag_exceptional_parent_child_relationship=False
                if tmp_dict_child_feature["contig"]!=tmp_dict_feature["contig"]:
                    flag_exceptional_parent_child_relationship=True
                elif child_location.start<location.start or child_location.end>location.end:
                    flag_exceptional_parent_child_relationship=True
                elif child_location.strand!=location.strand:
                    flag_exceptional_parent_child_relationship=True
                if flag_exceptional_parent_child_relationship:
                    a.append([feature_id,child_feature_id])
    n_inconsistent_parent_child_pair=len(a)
    if n_inconsistent_parent_child_pair>0:
        df_inconsistent_parent_child_pair=pd.DataFrame(a,columns=["parent_feature_id","child_feature_id"])
        gdf_inconsistent_parent_child_pair=df_inconsistent_parent_child_pair.groupby("parent_feature_id")
        flag_omitting=False
        l_exceptional_parent_feature_id=list(gdf_inconsistent_parent_child_pair.groups.keys())
        n_exceptional_parent_feature_id=len(l_exceptional_parent_feature_id)
        if n_exceptional_parent_feature_id>n_omitting_threshold:
            flag_omitting=True
            l_exhibited_exceptional_parent_feature_id=l_exceptional_parent_feature_id[:n_omitting_threshold]
        else:
            l_exhibited_exceptional_parent_feature_id=l_exceptional_parent_feature_id
        str_exception=""
        str_exception+="Identify {:d} feature(s) with inconsistent children features:".format(n_exceptional_parent_feature_id)+"\n\n"
        for i,exceptional_parent_feature_id in enumerate(l_exhibited_exceptional_parent_feature_id):
            tdf_inconsistent_parent_child_pair=gdf_inconsistent_parent_child_pair.get_group(exceptional_parent_feature_id)
            l_child_feature_id=tdf_inconsistent_parent_child_pair["child_feature_id"].values
            str_exception+="Parent feature ID: "+exceptional_parent_feature_id+'\n'
            str_exception+="Inconsistent child feature(s):"+'\n'
            b=[]
            for child_feature_id in l_child_feature_id:
                tmp_dict_child_feature=dict_feature[child_feature_id]
                b.append([tmp_dict_child_feature["contig"],tmp_dict_child_feature["feature_type"],str(tmp_dict_child_feature["location"]),str(tmp_dict_child_feature["dict_attribute"])])
            str_exception+=str(pd.DataFrame(b))
            if flag_omitting:
                str_exception+="\n\n\n"
            elif i!=len(l_exhibited_exceptional_parent_feature_id)-1:
                str_exception+="\n\n\n"
        if flag_omitting:
            str_exception+="......"
        raise Exception(str_exception)
    
    e_time=time.time()
    print("Finished, {:d}s taken. Identify {:d} sequence record(s) in total.".format(int(e_time-s_time),n_record))
    print()