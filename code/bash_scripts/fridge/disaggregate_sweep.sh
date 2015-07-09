#!/bin/sh


source ~/.bashrc

for ((N_STATES=2; N_STATES<3; N_STATES+=1))
    do
    for ((K=2;K<3;K+=1))
        do
        for((TRAIN=10;TRAIN<30;TRAIN+=10))
            do
            OFILE=../../../results/fridge/sweep_results/N${N_STATES}_K${K}_T${TRAIN}.out
            EFILE=../../../results/fridge/sweep_results/N${N_STATES}_K${K}_T${TRAIN}.err

            SLURM_SCRIPT=N${N_STATES}_K${K}_T${TRAIN}.pbs
            CMD='python ../../fridge/disaggregate.py ~/wikienergy-2.h5 '$N_STATES' '$K' '$TRAIN''
            echo $CMD

            #rm ${SLURM_SCRIPT}
            echo "#!/bin/sh" > ${SLURM_SCRIPT}
            #echo $pwd > ${SLURM_SCRIPT}
            echo '#SBATCH --time=02:0:00' >> ${SLURM_SCRIPT}
            echo '#SBATCH --mem=16' >> ${SLURM_SCRIPT}
            echo '#SBATCH -o "./'${OFILE}'"' >> ${SLURM_SCRIPT}
            echo '#SBATCH -e "./'${EFILE}'"' >> ${SLURM_SCRIPT}
            #echo 'cd $SLURM_SUBMIT_DIR' >> ${SLURM_SCRIPT}
            echo ${CMD} >> ${SLURM_SCRIPT}

            #cat ${SLURM_SCRIPT}
            sbatch ${SLURM_SCRIPT}
            done
        done
    done




