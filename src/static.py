str_version="V1.0.00"
str_logo=\
'''                 __         ___  ___ __  __ 
                / _|  ___  | _ )/ __|  \/  |
               |  _| |___| | _ \ (_ | |\/| |
               |_|         |___/\___|_|  |_|
'''

dict_suffix2file_type={
    "gbk":"genbank",
    "gb":"genbank",
    "gbff":"genbank",
    "fa":"fasta",
    "fasta":"fasta",
    "fna":"fasta",
    "gff":"gff",
    "gff3":"gff"
}

l_suffix_file_genome_sequence=["gbk","gb","gbff","fa","fasta","fna"]
set_suffix_file_genome_sequence={"gbk","gb","gbff","fa","fasta","fna"}
l_suffix_file_genome_annotation=["gff","gff3"]
set_suffix_file_genome_annotation={"gff","gff3"}


dict_status_code={
    0:"In Progress.",
    1:"Finished.",
    2:"Failed.",
}

dict_step_code={
    0:{
        "title":"Check the genome sequence file.",
        "description":"",
        "memo":"genbank input"
    },
    10:{
        "title":"Extract protein sequence(s).",
        "description":"",
        "memo":"genbank input"
    },
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    20:{
        "title":"Check the genome sequence file.",
        "description":"",
        "memo":"fasta + gff input"
    },
    30:{
        "title":"Check the genome annotation file.",
        "description":"",
        "memo":"fasta + gff input"
    },
    40:{
        "title":"Extract protein sequence(s).",
        "description":"",
        "memo":"fasta + gff input"
    },
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    50:{
        "title":"Check the genome sequence file.",
        "description":"",
        "memo":"only fasta input"
    },
    60:{
        "title":"Generate genome annotation",
        "description":"",
        "memo":"only fasta input"
    },
    70:{
        "title":"Check the genome annotation file.",
        "description":"",
        "memo":"only fasta input"
    },
    80:{
        "title":"Extract protein sequence(s).",
        "description":"",
        "memo":"only fasta input"
    },
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    90:{
        "title":"Generate Pfam annotation",
        "description":"",
        "memo":""
    },
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    110:{
        "title":"Generate ESM representation",
        "description":"",
        "memo":""
    },
    120:{
        "title":"Run f-BGM",
        "description":"",
        "memo":""
    },
    130:{
        "title":"Perform BGC prediction",
        "description":"",
        "memo":""
    },
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    250:{
        "title":"Generate prediction detail",
        "description":"",
        "memo":""
    },
}