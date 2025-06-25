from pygustus import augustus
import os
import sys
import time

def callAUGUSTUS(file_fasta,ofile_gff,ofile_err,species,n_thread):
    try:
        augustus.predict(file_fasta,species=species,outfile=ofile_gff,errfile=ofile_err,gff3=True,jobs=n_thread)
        if os.path.getsize(ofile_err)==0:
            return True
        else:
            return False
    except Exception as e:
        return False


def genGenomeAnnotation(file_fasta,ofile_gff,ofile_err,species,n_thread):
    
    print("Genome annotation file abscent, turn to generate genome annotation de novo by AUGUSTUS...")
    print()
    print("Specified reference species: {:s}.".format(species))
    print()
    s_time=time.time()
    sys.stdout=open(os.devnull,'w')
    flag=callAUGUSTUS(
        file_fasta=file_fasta,
        ofile_gff=ofile_gff,
        ofile_err=ofile_err,
        species=species,
        n_thread=n_thread
    )
    sys.stdout=sys.__stdout__
    if not flag:
        raise Exception("Failed to generate genome annotation. Please (1) check the reference species name and (2) see "+ofile_err+" for solutions.")
    e_time=time.time()
    print("Finished, {:d}s taken.".format(int(e_time-s_time)))
    print()
    