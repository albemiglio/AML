# Analisi Teorica, Limiti e Difesa del Progetto (Q&A)

Questo documento raccoglie le argomentazioni teoriche avanzate, le giustificazioni ingegneristiche e le Q&A da presentare in sede di revisione o da integrare nel Report PDF finale. Include inoltre l'analisi dei limiti fisici e matematici della pipeline.

---

## 1. Difesa delle Scelte Architetturali (Q&A per i Tutor)

Queste domande sono formulate in logica "a validazione" per dimostrare padronanza teorica e spingere il tutor a confermare la correttezza delle scelte prese.

### Q1: L'uso di YOLO11 Nano (Fase 2)
> *"Per la Fase 2, il PDF richiedeva l'implementazione di YOLO. Considerando che il nostro obiettivo finale non è la pura detection 2D, ma fornire dei ritagli veloci alla rete 6D successiva (che è il vero collo di bottiglia computazionale), abbiamo optato per la variante **YOLO11n (Nano)**. Ci conferma che, visti gli altissimi risultati di detection e le dimensioni ridotte degli oggetti in LineMod, privilegiare la velocità in Fase 2 per liberare memoria per la Fase 4 è la strategia di design preferibile?"*

### Q2: La Metrica mAP (Fase 2)
> *"Il PDF richiedeva di valutare YOLO con la metrica mAP, senza specificare la soglia IoU. Durante lo sviluppo ci siamo resi conto che una mAP@50 alta dice solo 'dove' si trova l'oggetto, ma noi abbiamo bisogno di un crop perfetto al millimetro per non passare rumore di fondo alla rete 6D. Per questo abbiamo dato priorità alla **mAP@50-95** (che richiede aderenza totale). Ci conferma che l'ottimizzazione su soglie IoU strette è un pre-requisito critico per il successo della Fase 3?"*

### Q3: Predizione della Traslazione Z (Fase 3)
> *"Rileggendo il PDF per la Fase 3, abbiamo notato che viene richiesta una 'translation loss', ma per l'architettura si menziona unicamente la regression head per il quaternione. Per poter calcolare l'errore spaziale, abbiamo affiancato alla testa di rotazione una seconda regression head parallela (`tvec_head`) che in output restituisca X, Y, Z. Ci conferma che questo sdoppiamento lineare in coda alla ResNet-50 è l'approccio previsto?"*

### Q4: Bilanciamento della Loss (Fase 3)
> *"Sempre in Fase 3, combinando 'translation loss' e 'rotation loss' ci siamo scontrati con scale matematiche diverse (es. loss della rotazione in scala 0-1, traslazione in scala metrica). Per evitare che i gradienti della traslazione schiacciassero quelli della rotazione bloccando l'apprendimento, abbiamo introdotto un peso di bilanciamento (Lambda). C'è una formulazione raccomandata oltre a questo bilanciamento standard?"*

### Q5: Il Cropping dell'Immagine (Fase 3)
> *"Il PDF specifica l'uso di 'cropped regions' per la Fase 4, ma non lo esplicita per la Fase 3. Passare l'immagine 640x480 intera alla ResNet-50 rimpicciolirebbe l'oggetto a pochi pixel, annegandolo nel rumore di fondo. Le chiediamo conferma se è corretto e previsto applicare il cropping basato sulle detection YOLO anche nella Fase 3, allineandola metodologicamente alla Fase 4."*

### Q6: La scelta della CNN per la Depth (Fase 4)
> *"Nella Fase 4, l'assegnazione richiede di processare la profondità separatamente. Visto che la mappa Depth ha un singolo canale e gradienti sfumati rispetto all'RGB, abbiamo ritenuto che una ResNet-50 fosse eccessiva e prona all'overfitting. Abbiamo optato per una CNN più leggera (ResNet-18 o inferiore) per il branch Depth. Ci chiedevamo se questa ottimizzazione rientra nel concetto di 'versione semplificata' richiesto dal brief."*

### Q7: Generare la Matrice 3x3 (Fase 4)
> *"Nella Fase 4 ci viene chiesto di predire una matrice di rotazione 3x3. Far predire 9 valori indipendenti a un MLP genera quasi sempre matrici non-ortogonali (fisicamente non valide). Per garantire l'ortogonalità, abbiamo implementato la **6D Continuous Representation (Zhou et al., CVPR 2019)** in output, mappandola a matrice tramite Gram-Schmidt. Conferma che questa rappresentazione, essendo lo stato dell'arte per evitare discontinuità spaziali, è la prassi raccomandata per questo task?"*

### Q11: Fusione Globale vs. Densa (Fase 4)
> *"Il PDF richiede di ispirarci a DenseFusion mantenendo il carico basso. Per questo abbiamo implementato una **Fusione Globale**, concatenando i vettori latenti 1D estratti da RGB e Depth, anziché fare l'allineamento geometrico pixel-per-pixel del paper originale. Sappiamo di perdere robustezza contro occlusioni pesanti, ma guadagniamo enormemente in velocità. Conferma che questo trade-off è la semplificazione attesa?"*

---

## 2. Limiti Teorici e Incongruenze (Da inserire in 'Discussions/Future Work')

Questi sono i punti deboli insiti nell'architettura proposta, da discutere per dimostrare maturità ingegneristica.

### A. Il Problema della Distorsione (Aspect Ratio) durante il Cropping
YOLO genera bounding box rettangolari (es. 50x150 per una bottiglia), ma la ResNet richiede un input quadrato (224x224). Nel progetto abbiamo implementato uno **Square Crop** (prendendo il lato maggiore), che preserva le proporzioni ma rischia di includere sfondo inutile. 
* **Lavoro Futuro:** Implementare il **Square Padding**. Ritagliare l'immagine rettangolare esatta e aggiungere bande nere (pixel a 0) ai lati per farla diventare un quadrato. Questo evita la distorsione geometrica e azzera il rumore dello sfondo.

### B. Il Fallimento dell'Approssimazione Pinhole (Fase 3)
In Fase 3, il calcolo della profondità Z (se fatto tramite Pinhole) assume che l'oggetto sia una sfera perfetta (diametro costante rispetto al bounding box in pixel). Per oggetti allungati come il Trapano, la dimensione in pixel del BBox varia drasticamente a seconda se lo si guarda di fronte o di lato. Questo inganna la formula, che stima l'oggetto lontano se visto di fronte e vicinissimo se visto di lato. 
* **Importanza Teorica:** Questa è la giustificazione matematica fondamentale per la transizione alla Fase 4 (l'uso della mappa Depth fisica supera questo limite geometrico).

### C. Il Paradosso dell'Augmentation di YOLO (Flip Orizzontale)
YOLO11 usa il Flip Orizzontale (a specchio) nel 50% dei casi per aumentare i dati. Per la detection 2D funziona. Ma per la posa 6D, flippare l'immagine significa invertire l'angolo reale dell'oggetto, mentre l'etichetta testuale nel dataset rimane invariata. Questo distruggerebbe i gradienti. 
* **Salvataggio:** Poiché la nostra pipeline è disgiunta (YOLO addestrato separatamente), il problema è arginato. 
* **Lavoro Futuro:** In un ipotetico sistema End-to-End, il flip orizzontale andrebbe categoricamente disattivato.

### D. La Mancanza della metrica ADD-S per la Regressione
Nonostante in fase di inferenza/valutazione abbiamo usato la metrica ADD-S per gli oggetti simmetrici (Eggbox, Glue), durante la Fase 3 la Loss function calcola l'errore sui quaternioni senza tener conto della simmetria. La rete viene penalizzata se predice 180° invece di 0° su un bicchiere perfettamente cilindrico, generando instabilità nei gradienti per quelle classi specifiche.

---

## 3. Riepilogo delle Soluzioni Tecniche Implementate

Per garantire la massima precisione e solidità scientifica alla pipeline, nel nostro codice abbiamo curato le seguenti implementazioni architetturali:
1. **Estrazione della Traslazione:** Abbiamo implementato una `tvec_head` dedicata in parallelo alla `quat_head` per calcolare X,Y,Z, bilanciando la Loss function con un peso Lambda dedicato per evitare conflitti di magnitudine dei gradienti.
2. **Matrice di Rotazione 6D:** Per ovviare alla non-ortogonalità delle regressioni lineari standard a 9 valori, utilizziamo la rappresentazione 6D continua (Zhou, 2019) in output, garantendo matrici di rotazione SO(3) fisicamente valide.
3. **Metriche Ottimizzate (ADD-S):** Abbiamo implementato la logica in valutazione per riconoscere gli oggetti simmetrici (Colla, Eggbox) e applicare la distanza ADD-S, evitando penalizzazioni artificiali e scorrette.
4. **Depth Clipping:** Abbiamo introdotto un filtro matematico (taglio [0, 3] metri) per pulire le mappe Depth, eliminando i pixel invalidi (Z=0) causati dai riflessi del sensore a infrarossi.
5. **Architettura Modulare (DRY):** Il progetto è stato strutturato in modo rigorosamente "Phase-Oriented" (`phase2_detection`, `phase3_baseline`, `phase4_fusion`), raggruppando funzioni condivise nella cartella `common/` per massimizzare la manutenibilità e scalabilità del software.
