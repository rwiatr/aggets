#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J trainAggets
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=16GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=12:00:00 
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A plgmpragh
## Specyfikacja partycji
#SBATCH --partition=plgrid-gpu
#SBATCH --gres=gpu:1
## Plik ze standardowym wyjściem
#SBATCH --output="aggets.out"
## Plik ze standardowym wyjściem błędów
#SBATCH --error="aggets.err"

## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR

srun /bin/hostname
#PYTHONPATH="${PYTHONPATH}:~/aggets"

source ../venv/bin/activate

start=`date +%s.%N`
python3 main.py
end=`date +%s.%N`
runtime=$( echo "$end - $start" | bc -l | awk '{printf "%.8f\n", $0}')
echo "$end - $start"
echo "$runtime"
