File CSV prodotti mediante l'esecuzione di notebooks su jupyter.
Tali file sono stati adoperati per la produzione di grafici e tabelle presenti all'interno della presentazione PowerPoint.


Fissato m=64, n=16, nUNet=2, la nomenclatura segue il seguente criterio:


restart
Activation: identity, Optimizer: RMSProp , LR = 2.5e-4 , Kernel-Size del layer convoluzionale iniziale: 7x7 
------------
restartSig
Activation: sigmoid, Optimizer: RMSProp , LR = 2.5e-4 , Kernel-Size del layer convoluzionale iniziale: 7x7 
------------
restartSig2
Activation: sigmoid, Optimizer: RMSProp , LR = 2.5e-3 , Kernel-Size del layer convoluzionale iniziale: 7x7 
------------
restartSig3
Activation: sigmoid, Optimizer: RMSProp , LR = 6.7e-3 , Kernel-Size del layer convoluzionale iniziale: 7x7 
------------
restartSig4
Activation: sigmoid, Optimizer: RMSProp , LR = 6.7e-3 , Kernel-Size del layer convoluzionale iniziale: 7x7 , fattore di riduzione del lr: 0.6
------------
restartSig5
Activation: sigmoid, Optimizer: RMSProp , LR = 2.5e-3 , Kernel-Size del layer convoluzionale iniziale: 7x7 , heatmap target modificate
------------
restartSig6
Activation: sigmoid, Optimizer: RMSProp , LR = 2.5e-3 , Kernel-Size del layer convoluzionale iniziale: 5x5 + Conv2d 3x3 , heatmap target modificate
------------
restartSig7
Activation: sigmoid, Optimizer: RMSProp , LR = 6.7e-3 , Kernel-Size del layer convoluzionale iniziale: 5x5 + Conv2d 3x3 , heatmap target modificate
------------
restartSig8
Activation: sigmoid, Optimizer: Adam , LR = 6.7e-3 , Kernel-Size del layer convoluzionale iniziale: 5x5 + Conv2d 3x3 , heatmap target modificate


restartSig8 modello migliore.

A partire da questo modello, sono stati addestrati altri modelli variando m, n e numero di supervisioni.

