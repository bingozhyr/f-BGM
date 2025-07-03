import os
current_path=os.path.abspath(os.path.dirname(__file__))+"/"
import sys
from src import static
import argparse
from src.util import define_argument
import numpy as np
        
parser=argparse.ArgumentParser(add_help=False)

argument_group_io=parser.add_argument_group("Basic arguments")
argument_group_io.add_argument("-s","--sequence",type=str,required=True,help="Genome sequence file, genbank (*.gbk, *.gb and *.gbff) and fasta (*.fa, *.fasta and *.fna) format supported.")
argument_group_io.add_argument("-a","--annotation",type=str,default=None,help="Genome annotation file, gff3 (*.gff and *.gff3) format supported. "+\
                               "If the genome sequence file is of fasta format, then this argument will be used to extract gene and peptide information(if specified). "+\
                               "If the genome sequence file is of fasta format but this argument is not specified, then the AUGUSTUS tool will be called to generate genome annotation de novo. "+\
                               "If the genome sequence file is of genbank format, then this argument will be ignored. ")
argument_group_io.add_argument("-p","--path",type=str,required=True,help="Working path for the task, both temporary and final results will be saved here.")


argument_group_prediction=parser.add_argument_group("Prediction arguments")
argument_group_prediction.add_argument("--pred_score_threshold",type=define_argument.define_numeric_range("float",0,1),default=None,help="Peptide(s) whose prediction score(s) exceeding the threshold will be over-represented as BGC member(s). "+\
                                       "Only one of the parameters pred_score_threshold and pred_score_top_ratio is required. Valid range: [0,1]. Default: None")
argument_group_prediction.add_argument("--pred_score_top_ratio",type=define_argument.define_numeric_range("float",0,1),default=None,help="Peptide(s) whose prediction score(s) ranking in this top ratio will be over-represented as BGC member(s). "+\
                                       "Only one of the parameters pred_score_threshold and pred_score_top_ratio is required. Valid range: [0,1]. Default: None")
argument_group_prediction.add_argument("--pred_score_threshold_core_enzyme",type=define_argument.define_key_value_pairs(),default="PKS=0.98 NRPS=0.98 TC=0.76 PT=0.97 PKS-NRPS=0.98 TS-chimeric=0.97 PPPS=0.83",help="Prediction score threshold(s) for identifying specific core enzymes within putative BGC(s). "+\
                                       'Format: "key1=value1 key2=value2 key3=value3 ...". Default: "PKS=0.98 NRPS=0.98 TC=0.76 PT=0.97 PKS-NRPS=0.98 TS-chimeric=0.97 PPPS=0.83"')
argument_group_prediction.add_argument("--n_submodel",type=define_argument.define_numeric_range("int",1,16),default=16,help="Submodel number for prediction, a larger number will enhance prediction robustness. Valid range: [1,16]. Default: 16")
argument_group_prediction.add_argument("--min_n_pep_contig",type=define_argument.define_numeric_range("int",1,65535),default=32,help="Contig(s) whose peptide(s) less than this threshold will be ignored. Valid range: [1,65535]. Default: 32")
argument_group_prediction.add_argument("--min_n_pep_bgc",type=define_argument.define_numeric_range("int",1,65535),default=1,help="Putative BGC(s) whose member peptide(s) less than this threshold will be ignored. Valid range: [1,65535]. Default: 1")
argument_group_prediction.add_argument("--max_n_pep_bgc",type=define_argument.define_numeric_range("int",1,65535),default=128,help="Putative BGC(s) whose member peptide(s) more than this threshold will be ignored. Valid range: [1,65535]. Default: 128")
argument_group_prediction.add_argument("--max_interval_n_pep_bgc_merging",type=define_argument.define_numeric_range("int",0,65535),default=1,help="Any two putative BGCs whose interval peptide number no more than this threshold will be merged. Valid range: [0,65535]. Default: 1")
argument_group_prediction.add_argument("--flanking_dna_sequence_length",type=define_argument.define_numeric_range("int",0,2147483647),default=0,help="Base pair(bp) number extended in both 5'- and 3'-side when reporting DNA sequence(s) of putative BGC(s). Valid range: [0,2147483647]. Default: 0")
argument_group_prediction.add_argument("--device_type",type=str,choices=["cpu","gpu"],default=None,help="Device type for f-BGM running, if this argument is abscent, then the program will automatically select an available device (GPU device preferred if exists).")


argument_group_augustus=parser.add_argument_group("AUGUSTUS arguments (used to generate genome annotation when the genome sequence file is of fasta format but the genome annotation file is abscent)")
argument_group_augustus.add_argument("--augustus_species",type=str,default="anidulans",help="Reference species (built-in the AUGUSTUS tool) used to generate genome annotation, see 'augustus --species=help' for detail. Default: anidulans")
argument_group_augustus.add_argument("--n_thread_augustus",type=define_argument.define_numeric_range("int",1,np.inf),default=1,help="CPU number used to generate genome annotation using AUGUSTUS. Default: 1")


argument_group_hmmer=parser.add_argument_group("HMMER arguments (used to generate Pfam domain annotation for peptide sequence(s))")
argument_group_hmmer.add_argument("--e_threshold_hmmer",type=define_argument.define_numeric_range("float",0,1),default=0.01,help="E-value cutoff used to define Pfam domain(s) with statistical significance. Default: 0.01")
argument_group_hmmer.add_argument("--n_thread_hmmer",type=define_argument.define_numeric_range("int",1,np.inf),default=1,help="CPU number used to to generate Pfam domain annotation by HMMER. Default: 1")

argument_group_other=parser.add_argument_group("Other arguments")
argument_group_other.add_argument("-h","--help",action="help",help="Print help message.")
argument_group_other.add_argument("-v","--version",action="version",version=static.str_version,help="Print version information.")

args=parser.parse_args()


file_genome_sequence=args.sequence
file_genome_annotation=args.annotation
path_task=args.path

pred_score_threshold=args.pred_score_threshold
pred_score_top_ratio=args.pred_score_top_ratio
dict_pred_score_threshold_core_enzyme=args.pred_score_threshold_core_enzyme
n_submodel=args.n_submodel
min_n_pep_contig=args.min_n_pep_contig
min_n_pep_BGC=args.min_n_pep_bgc
max_n_pep_BGC=args.max_n_pep_bgc
max_interval_n_pep_BGC_merging=args.max_interval_n_pep_bgc_merging
flanking_DNA_sequence_length=args.flanking_dna_sequence_length
device_type=args.device_type

augustus_species=args.augustus_species
n_thread_augustus=args.n_thread_augustus

n_thread_hmmer=args.n_thread_hmmer
e_threshold_hmmer=args.e_threshold_hmmer

path_model=current_path+"model/f-BGM/"

ref_file_pfam_A_dat=current_path+"external_file/Pfam-A.hmm.dat"
ref_file_pfam_A=current_path+"external_file/Pfam-A.hmm"

from src import static_config
import pickle
import json
import time
from src.util import calculate_file_md5,check_environment,get_available_device
from src.input_processing import generate_genome_annotation,precheck,extract_pep_seq,generate_pfam_annotation
from src.model.fbgm import generate_fesm2_representation,genome_mining,bgc_prediction
from src.result_analysis import generate_prediction_detail



dict_suffix2file_type=static.dict_suffix2file_type
l_suffix_file_genome_sequence=static.l_suffix_file_genome_sequence
l_suffix_file_genome_annotation=static.l_suffix_file_genome_annotation
set_suffix_file_genome_sequence=static.set_suffix_file_genome_sequence
set_suffix_file_genome_annotation=static.set_suffix_file_genome_annotation
dict_step_code=static.dict_step_code
dict_status_code=static.dict_status_code

print()
print("= = = = = = = = = = = = = = = = = = = = = = = = = = = = = =")
print(static.str_logo)
print("= = = = = = = = = = = = = "+static.str_version+" = = = = = = = = = = = = =")
print()

flag_pred_score_threshold,flag_pred_score_top_ratio=False,False
if pred_score_threshold is not None:
    flag_pred_score_threshold=True
if pred_score_top_ratio is not None:
    flag_pred_score_top_ratio=True
if flag_pred_score_threshold and flag_pred_score_top_ratio:
    raise Exception("Only one of the parameters pred_score_threshold and pred_score_top_ratio is required.")
if not flag_pred_score_threshold and not flag_pred_score_top_ratio:
    raise Exception("One of the parameters pred_score_threshold and pred_score_top_ratio should be specified.")
    
suffix_file_genome_sequence=file_genome_sequence.split('.')[-1]
if suffix_file_genome_sequence not in set_suffix_file_genome_sequence:
    raise Exception("Unsupported genome sequence file type, "+', '.join(["*."+suffix for suffix in l_suffix_file_genome_sequence])+" file expected.")

if file_genome_annotation is not None:
    suffix_file_genome_annotation=file_genome_annotation.split('.')[-1]
    if suffix_file_genome_annotation not in set_suffix_file_genome_annotation:
        raise Exception("Unsupported genome annotation file type, "+', '.join(["*."+suffix for suffix in l_suffix_file_genome_annotation])+" file expected.")

file_type_genome_sequence=dict_suffix2file_type[suffix_file_genome_sequence]
file_type_genome_annotation=dict_suffix2file_type[suffix_file_genome_annotation] if file_genome_annotation is not None else None
        
md5_file_genome_sequence=calculate_file_md5.calFileMD5(file_genome_sequence)
md5_file_genome_annotation=calculate_file_md5.calFileMD5(file_genome_annotation) if file_genome_annotation is not None else None

path_task=os.path.abspath(path_task)+'/'
if not os.path.exists(path_task):
    os.makedirs(path_task)
path_tmp_result=path_task+"tmp_result/"

flag_new_task=False
file_log=path_task+"log.pkl"
if os.path.exists(file_log):
    with open(file_log,"rb") as f:
        dict_log=pickle.load(f)
    if dict_log["file_md5sum"]!=(md5_file_genome_sequence,md5_file_genome_annotation):
        flag_new_task=True
    elif file_type_genome_sequence=="fasta" and file_type_genome_annotation is None and augustus_species!=dict_log["mode_augustus"]:
        flag_new_task=True
    else:
        time_stamp=str(int(time.time()))

        if (110,1) not in dict_log["set_step"]:
            if len(os.listdir(dict_log["path_tmp_result_fbgm"]))!=0:
                dict_log["path_tmp_result_fbgm"]=path_tmp_result+"fbgm_tmp_result/"+time_stamp+'/'
                os.makedirs(dict_log["path_tmp_result_fbgm"])
                    
        dict_log["path_prediction_result"]=path_task+"prediction_result/"+time_stamp+'/'
        os.makedirs(dict_log["path_prediction_result"])
            
        dict_log["path_prediction_result_BGC_detail"]=dict_log["path_prediction_result"]+"putative_BGC_detail/"
        os.makedirs(dict_log["path_prediction_result_BGC_detail"])
            
        dict_log["path_prediction_result_contig_detail"]=dict_log["path_prediction_result"]+"contig_detail/"
        os.makedirs(dict_log["path_prediction_result_contig_detail"])
else:
    flag_new_task=True
    
if flag_new_task:
    dict_log={}
    time_stamp=str(int(time.time()))
    dict_log["file_md5sum"]=(md5_file_genome_sequence,md5_file_genome_annotation)
    dict_log["path_seq"]=path_task+"seq/"+time_stamp+'/'
    os.makedirs(dict_log["path_seq"])

    dict_log["path_tmp_result_fbgm"]=path_tmp_result+"fbgm_tmp_result/"+time_stamp+'/'
    os.makedirs(dict_log["path_tmp_result_fbgm"])
        
    dict_log["path_tmp_result_genome_mining"]=path_tmp_result+"genome_mining_tmp_result/"+time_stamp+'/'
    os.makedirs(dict_log["path_tmp_result_genome_mining"])
        
    dict_log["path_prediction_result"]=path_task+"prediction_result/"+time_stamp+'/'
    os.makedirs(dict_log["path_prediction_result"])
        
    dict_log["path_prediction_result_BGC_detail"]=dict_log["path_prediction_result"]+"putative_BGC_detail/"
    os.makedirs(dict_log["path_prediction_result_BGC_detail"])
        
    dict_log["path_prediction_result_contig_detail"]=dict_log["path_prediction_result"]+"contig_detail/"
    os.makedirs(dict_log["path_prediction_result_contig_detail"])
    
    if file_type_genome_sequence=="fasta" and file_type_genome_annotation is None:
        dict_log["mode_augustus"]=augustus_species
    else:
        dict_log["mode_augustus"]=None
    
    dict_log["set_step"]=set()
    
with open(file_log,"wb") as f:
    pickle.dump(dict_log,f)

try:
    if (file_type_genome_sequence=="fasta" and file_type_genome_annotation is None and (60,1) not in dict_log["set_step"]):
        
        print("Checking the base environment...")
        print()
        path_base_env=check_environment.getPathBaseEnv()
        if path_base_env is None:
            raise Exception("Failed to locate the base environment.")
        print("Finished. The base environment is located in {:s}.".format(path_base_env))
        print()
        print()
    
    if file_type_genome_sequence=="genbank":
        step_code=0
        if (step_code,1) not in dict_log["set_step"]:
            dict_log["set_step"].add((step_code,0))
            precheck.checkGenBankFile(file=file_genome_sequence)
            dict_log["set_step"].add((step_code,1))
            print()
    
        step_code=10
        if (step_code,1) not in dict_log["set_step"]:
            dict_log["set_step"].add((step_code,0))
            extract_pep_seq.extractPepSeqFromGenBank(
                file_genbank=file_genome_sequence,
                ofile_contig_fasta=dict_log["path_seq"]+"contig.fasta",
                ofile_df_pep_seq=dict_log["path_seq"]+"pep_seq.csv",
                ofile_contig_list=dict_log["path_seq"]+"contig_list.npy",
                ofile_fasta_pep_seq=dict_log["path_seq"]+"pep_seq.fasta",opath_seq=dict_log["path_seq"]
            )
            dict_log["set_step"].add((step_code,1))
            print()
    
    elif file_type_genome_sequence=="fasta":
        if file_type_genome_annotation=="gff":
    
            step_code=20
            if (step_code,1) not in dict_log["set_step"]:
                dict_log["set_step"].add((step_code,0))
                precheck.checkFASTAFile(file=file_genome_sequence,sequence_type="nucleotide")
                dict_log["set_step"].add((step_code,1))
                print()
    
            step_code=30
            if (step_code,1) not in dict_log["set_step"]:
                dict_log["set_step"].add((step_code,0))
                precheck.checkGenomeAnnotationFile(file=file_genome_annotation,file_type=file_type_genome_annotation,ref_file_fasta=file_genome_sequence)
                dict_log["set_step"].add((step_code,1))
                print()
    
            step_code=40
            if (step_code,1) not in dict_log["set_step"]:
                dict_log["set_step"].add((step_code,0))
                extract_pep_seq.extractPepSeqFromFASTA(
                    file_fasta=file_genome_sequence,
                    file_genome_annotation=file_genome_annotation,
                    file_type_genome_annotation=file_type_genome_annotation,
                    ofile_contig_fasta=dict_log["path_seq"]+"contig.fasta",
                    ofile_df_pep_seq=dict_log["path_seq"]+"pep_seq.csv",
                    ofile_contig_list=dict_log["path_seq"]+"contig_list.npy",
                    ofile_fasta_pep_seq=dict_log["path_seq"]+"pep_seq.fasta",opath_seq=dict_log["path_seq"]
                )
                dict_log["set_step"].add((step_code,1))
                print()
    
        elif file_type_genome_annotation is None:
            file_genome_annotation=dict_log["path_seq"]+"augustus.gff"
            file_augustus_err=dict_log["path_seq"]+"augustus.err"
            file_type_genome_annotation="gff"
    
            step_code=50
            if (step_code,1) not in dict_log["set_step"]:
                dict_log["set_step"].add((step_code,0))
                precheck.checkFASTAFile(file=file_genome_sequence,sequence_type="nucleotide")
                dict_log["set_step"].add((step_code,1))
                print()
    
            step_code=60
            if (step_code,1) not in dict_log["set_step"]:
                dict_log["set_step"].add((step_code,0))
                generate_genome_annotation.genGenomeAnnotation(
                    file_fasta=file_genome_sequence,
                    ofile_gff=file_genome_annotation,
                    ofile_err=file_augustus_err,
                    species=augustus_species,
                    n_thread=n_thread_augustus
                )
                dict_log["set_step"].add((step_code,1))
                print()
    
            step_code=70
            if (step_code,1) not in dict_log["set_step"]:
                dict_log["set_step"].add((step_code,0))
                precheck.checkGenomeAnnotationFile(file=file_genome_annotation,file_type=file_type_genome_annotation,ref_file_fasta=file_genome_sequence)
                dict_log["set_step"].add((step_code,1))
                print()
    
            step_code=80
            if (step_code,1) not in dict_log["set_step"]:
                dict_log["set_step"].add((step_code,0))
                extract_pep_seq.extractPepSeqFromFASTA(
                    file_fasta=file_genome_sequence,
                    file_genome_annotation=file_genome_annotation,
                    file_type_genome_annotation=file_type_genome_annotation,
                    ofile_contig_fasta=dict_log["path_seq"]+"contig.fasta",
                    ofile_df_pep_seq=dict_log["path_seq"]+"pep_seq.csv",
                    ofile_contig_list=dict_log["path_seq"]+"contig_list.npy",
                    ofile_fasta_pep_seq=dict_log["path_seq"]+"pep_seq.fasta",opath_seq=dict_log["path_seq"]
                )
                dict_log["set_step"].add((step_code,1))
                print()
    
    
    step_code=90
    if (step_code,1) not in dict_log["set_step"]:
        dict_log["set_step"].add((step_code,0))
        generate_pfam_annotation.genPfamAnnotation(
            file_contig_list=dict_log["path_seq"]+"contig_list.npy",
            file_fasta_pep_seq=dict_log["path_seq"]+"pep_seq.fasta",
            path_seq=dict_log["path_seq"],
            ofile_pfam_json=dict_log["path_seq"]+"pep_seq.pfam.json",
            ref_file_pfam_A=ref_file_pfam_A,
            n_thread=n_thread_hmmer,
            e_threshold=e_threshold_hmmer
        )
        dict_log["set_step"].add((step_code,1))
        print()
    
    
    print("Selecting device for model running...")
    print()
    selected_device_tuple=get_available_device.getAvailableDevice(required_memory=static_config.required_memory,min_retained_gpu_memory=static_config.min_retained_gpu_memory,min_retained_system_memory=static_config.min_retained_system_memory,device_type=device_type)
    if selected_device_tuple is None:
        raise Exception("No free device found.")
    selected_device_type,selected_device_index,selected_device=selected_device_tuple
    if selected_device_type=="gpu":
        print("Finished. Selected device type: {:s}, index: {:d}.".format(selected_device_type,selected_device_index))
    elif selected_device_type=="cpu":
        print("Finished. Selected device type: {:s}.".format(selected_device_type))
    print()
    print()
    
    
    step_code=110
    if (step_code,1) not in dict_log["set_step"]:
        dict_log["set_step"].add((step_code,0))
        generate_fesm2_representation.genfESM2Representation(
            file_contig_list=dict_log["path_seq"]+"contig_list.npy",
            path_seq=dict_log["path_seq"],
            opath_esm_json=dict_log["path_tmp_result_fbgm"],
            path_model=path_model,
            device=selected_device
        )
        dict_log["set_step"].add((step_code,1))
        print()
    
    step_code=120
    if (step_code,1) not in dict_log["set_step"]:
        dict_log["set_step"].add((step_code,0))
        genome_mining.performGenomeMining(
            file_contig_list=dict_log["path_seq"]+"contig_list.npy",
            path_seq=dict_log["path_seq"],
            file_pfam_json=dict_log["path_seq"]+"pep_seq.pfam.json",
            path_esm_json=dict_log["path_tmp_result_fbgm"],
            file_df_pep_seq=dict_log["path_seq"]+"pep_seq.csv",
            opath_genome_mining_result=dict_log["path_tmp_result_genome_mining"],
            path_model=path_model,
            n_submodel=n_submodel,
            device=selected_device
        )
        dict_log["set_step"].add((step_code,1))
        print()
    
    # step_code=130
    # if (step_code,1) not in dict_log["set_step"]:
    #     dict_log["set_step"].add((step_code,0))
    pred_score_threshold_=bgc_prediction.performBGCPrediction(
        file_contig_list=dict_log["path_seq"]+"contig_list.npy",
        file_pfam_json=dict_log["path_seq"]+"pep_seq.pfam.json",
        path_esm_json=dict_log["path_tmp_result_fbgm"],
        path_genome_mining_result=dict_log["path_tmp_result_genome_mining"],
        ref_file_pfam_A_dat=ref_file_pfam_A_dat,
        ofile_df_putative_BGC=dict_log["path_prediction_result"]+"putative_BGC.csv",
        min_n_pep_contig=min_n_pep_contig,
        min_n_pep_BGC=min_n_pep_BGC,
        max_n_pep_BGC=max_n_pep_BGC,
        max_interval_n_pep_BGC_merging=max_interval_n_pep_BGC_merging,
        pred_score_threshold=pred_score_threshold,
        pred_score_top_ratio=pred_score_top_ratio,
        dict_pred_score_threshold_core_enzyme=dict_pred_score_threshold_core_enzyme,
        path_model=path_model,
        n_submodel=n_submodel,
        device=selected_device
    )
    # dict_log["set_step"].add((step_code,1))
    print()
    print()
    
    generate_prediction_detail.genPredictionDetail(
        file_contig_list=dict_log["path_seq"]+"contig_list.npy",
        file_contig_fasta=dict_log["path_seq"]+"contig.fasta",
        file_pfam_json=dict_log["path_seq"]+"pep_seq.pfam.json",
        path_esm_json=dict_log["path_tmp_result_fbgm"],
        path_genome_mining_result=dict_log["path_tmp_result_genome_mining"],
        path_prediction_result=dict_log["path_prediction_result"],
        opath_prediction_result_BGC_detail=dict_log["path_prediction_result_BGC_detail"],
        opath_prediction_result_contig_detail=dict_log["path_prediction_result_contig_detail"],
        path_model=path_model,
        ref_file_pfam_A_dat=ref_file_pfam_A_dat,
        flag_pred_score_threshold=flag_pred_score_threshold,
        pred_score_threshold=pred_score_threshold_,
        flanking_DNA_sequence_length=flanking_DNA_sequence_length,
        n_submodel=n_submodel,
        device=selected_device
    )

except Exception as e:
    print("Exception:",e)
    print()
    
finally:
    with open(file_log,"wb") as f:
        pickle.dump(dict_log,f)
