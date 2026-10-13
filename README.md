This project aimed to use a deep learning model to distinguish between Channel State Information (CSI) data of 12 classes of actions or movements. 
All data were collected in the same room with the same layout, over multiple days.
This was done in the Industrial Complex of Hong Kong Polytechnic University, from July to September of 2026. 

The CSI data collected was preprocessed, and then made into many smaller windows of 4000 packets with 2000 window stride. 

The Model folder contains the models of the epoch of best validation f1 score, as well as the final outputs in the testing stage.
Here is a quick overview of the model files:
- 7 Classes: Early on, we collected data for Empty, Sitting, Standing, Normal Walking, Slow Walking, Standing Up, and Sitting Down. The two models in the '7 Classes' folder were trained strictly on these initial 7 classes.
- All Classes: We then collected the remaining 5 classes (Falling, Reaching Out, Bending and Recovering, Lying Down, and Getting Up). The four models in the 'All classes' folder are trained on the complete 12-class dataset. There are two models that were tagged with the 'OLD' prefix which were based on old datasets; I have included them in case they are helpful for comparison.

Overall, the model shows solid performance, though there are a few specific classes it currently struggles to distinguish cleanly.
The main confusion occurs between two sets of classes: Sitting/Standing versus Standing Up/Sitting Down. This is because they share very similar positional data and only differ in small movement fluctuations.

The 7 python scripts used are in the Python folder, and their uses are listed below:
- clean_nexmon_csi.py: Visualizes the data, as well as removes packet bursting by linear interpolation.
- build_fixed_packet_segments.py: Creates sliding window packets to slice long pcap files into smaller training inputs.
- model_training.py: Handles deep learning model training.
- read_confusion_matrix.py: Prints out the .npy confusion matrix files.
- test_model.py: Tests an output model.
- score_60s_window.py & select_clean_60s_from_tail.py: Written when we initially tried using amplitude-against-time data to select 60-second training windows. We later switched to standard packet data because it was simpler to extract, easier to read, and avoided issues like large empty gaps in time.

The raw data collected, as well as the processed data, was too large to be put on github. Instead, it is hosted on Hugging Face: https://huggingface.co/datasets/CrystalStarburst/CSIdata/tree/main
There it contains 3 folders:
- Raw_Data_by_Class: Raw CSI data in pcap format
- fixed_packet_segments_w4000_s2000: Packet segments of all classes, with window width 4000 packets and window stride 2000 packets
- split_packet_segments: Packet segments split into training, validation, and test. This is the folder that is to be inputted into the model_training.py code
