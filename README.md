# Road-Entities-Classifier
A Deep Learning model for classifying on-road entities using CNN with Progressive Resizing.
Please refer the report for methodology.

User Guide


1.	Creating dataset
		•	Please see sec.2.1 in report for diagrams. 
		•	Querying the search engine for the required class labels
				The required class label can be queried on images.google.com. 
				If majority of the results are satisfactory, proceed. Else a different query can be fired. 
		•	Fetching the URL’s of resulting images through JavaScript 
				The TXT file containing the URL’s of the above search results can be downloaded by executing the provided “js_console.js” file, step by step in the JavaScript console of the web browser (under the “developer” menu-bar item of the browser). A file named “urls.txt” gets downloaded at this step.
		•	Downloading the images through python script
				This “urls.txt” file should be placed in the directory where the “download_images.py” script file is present. In terminal, execute the head command to see the format for providing the “urls.txt” file & destination directory path to this script. Download script should be executed in the following way: 
				Execute –	python3 download_images.py –urls urls.txt --output path/to/dir

2.	Choosing a predefined dataset
		•	Make sure the class labels are same as (bike,bus,car,pedestrian).
		•	The dataset should be arranged in the hierarchical structure as explained in section 2.2. E.g.
				data/train/bus
				data/train/car
				data/val/pedestrian
				data/test/bus
				data/test/bike

		•	A sample dataset can be downloaded from -  
				https://drive.google.com/file/d/1yAvTpaNdeqVqi7TPo1FHkMltU2wM82ns/view?usp=sharing

3.	Training
		•	To train the model, place the “train.py” file in the same directory as dataset’s “data” folder.
		•	Execute – python3 train.py

4.	Testing
		•	To test the model, place the “test.py” file in the same directory as dataset’s “data” folder.
		•	Make sure the weight files, created by the training script are present in the same directory.
		•	Execute – python3 test.py




Dependencies - 
	imutils
	argparse
	requests
	opencv
	numpy
	pandas
	matplotlib
	sklearn
	tensorflow
	keras
Please check versions in requirements.txt
