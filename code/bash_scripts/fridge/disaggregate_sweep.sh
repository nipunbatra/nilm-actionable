#!/bin/sh


source ~/.bashrc

for ((N_STATES=2; N_STATES<5; N_STATES+=1))
    do
    for ((K=2;K<8;K+=1))
        do

        OFILE=../../../results/fridge/sweep_results/N${N_STATES}_K${K}.out
        EFILE=../../../results/fridge/sweep_results/N${N_STATES}_K${K}.err

        SLURM_SCRIPT=N${N_STATES}_K${K}.pbs
        CMD='python ../../fridge/test.py '$N_STATES' '$K''
        echo $CMD

        rm ${SLURM_SCRIPT}
        echo "#!/bin/sh" > ${SLURM_SCRIPT}
        echo $pwd > ${SLURM_SCRIPT}
        echo '#SBATCH --time=4-00:0:00' >> ${SLURM_SCRIPT}
        echo '#SBATCH --mem=4' >> ${SLURM_SCRIPT}
        echo '#SBATCH -o "./'${OFILE}'"' >> ${SLURM_SCRIPT}
        echo '#SBATCH -e "./'${EFILE}'"' >> ${SLURM_SCRIPT}
        #echo 'cd $SLURM_SUBMIT_DIR' >> ${SLURM_SCRIPT}
        echo ${CMD} >> ${SLURM_SCRIPT}

        #cat ${SLURM_SCRIPT}
        sbatch ${SLURM_SCRIPT}

        done
    done




