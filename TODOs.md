## CURRENT:  Refinement classifier 
### Done:
#### Training data selection
#### Trainnig data creation with old classifier
#### Training vector creation
#### Training f
###Doing
####Refinment Prediciton
####1: predict labels using "normal classifier"  
  
  -  save as normal labels in label folder  
 
  - save link in label-filelist normally

####2: read feature vectors from those labels

   

####3: predict labels again

- save as refined labels (X+"_refined.nc")
- save links also in label file list


## Next Steps

### Structure

    - Maybe: move all project parameters in own class  **parameters.py**
    - simplify inheritance and class strucures
    - make all filepaths sttable in json file
    - reduce class variables
    
    Reuduce data_handler/cc complexity by outsourcing into other files?
    - auslagern von funktionalität:
        - plotting (dh)

    - eval und plot settings in eigene file! parameter settings
    - pipeline settings?
    
###Architecture:

    
    - Python paket
        - dependencies
        - os-compatability



### Functionality:

    create evaluation set from folder!

    create inputset from folder
### Training:


    - Neue Classifier:
        - support vector machine (SVM)W
        - (simple) neural networks (NN)

    - Datenerweiterung:
        - clear-sky-data
        - verwendung von räumlich/zeitlich benachbarten datenpunkten

    - Maybe later:
        - verwendung großer bildausschnitte (mit CNN classifier)




    ### Documentation:

    - docstring vervollständigen
    - readme



## Done:



    - Basis-Klasse mit Vererbung um gemeinsame funktionalität des parameter setzen zum abkaplseln und verfeinern
        - i.e: Nur not-None argumente werden übernommen 
        - **kwargs um variabeln freier zuzuordene
        - verbose meldung von falsch übergeben variablen

    - Wrapper Class
        - einlesen von parametern (json?)
        - schreiben vom parametern
        - methoden um pipeline als ganzes durchzulaufen/ teilschritte zu bearbeiten
    
    - automatic version recognition

    - extrahieren via folder_location
        - automatic name parsing and mapping off corresponding sat and label files

    - fileliest in eigen ordner? (braucht anpassung base class)

    - handling of external georefs

    - auslagern von funktionalität:
        - filelist creation (cc)

    Plotting:
        - automatic version recognition