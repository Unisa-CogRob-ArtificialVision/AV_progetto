Inserire in questa cartella il training/validation set con le annotazioni.
In particolare:
- Inserire le immagini di training in una cartella "training_set"
- Inserire le annotazioni di training in "training_set.txt"
- Inserire le immagini di validation in una cartella "validation_set"
- Inserire le annotazioni di validation in "validation_set.txt"

Per addestrare chiamare il modulo "par_train_custom.py"
Per far partire il test del timing dei vari modelli eseguire "par_time_test.py"

par_model.py serve per costruire il modello per effettuare il PAR. Viene usato nello script principale "group09.py"