Eseguire: group09.py --video "path_to_video.mp4" --configuration "path_to_config.txt" --results "path_to_results.txt"

- Deepsort_yolov8.py contiene la classe del tracker che utilizza come algoritmo DeepSort. E' la classe attualmente usata.
- yolov8_tracker.py è un'implementazione del tracker utilizzando come algoritmo di tracking botsort. Non è utilizzata al momento (è stata una prova).
- DETECTOR_MODELS contiene il modello che effettua la detection
- PAR contiene tutto ciò che riguarda il PAR, cioè i modelli per le reti single task e il codice per il training
- evaluate_motchallenge.py crea il file di tracking valutando il tracker sul dataset MOT16 (dati di train) (usa deeprsort_app.py)

