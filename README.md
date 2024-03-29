# Non-Binary Bottom-up Shift-Reduce Constituent Parser
This repository includes the code of the non-binary bottom-up parser described in the paper [Faster Shift-Reduce Constituent Parsing with a Non-Binary, Bottom-up Strategy](http://doi.org/10.1007/s10462-017-9584-0) published in [Artificial Intelligence](https://www.sciencedirect.com/journal/artificial-intelligence) . The implementation is based on this framework (https://github.com/LeonCrashCode/InOrderParser) and reuses part of its code, including data preparation and evaluating scripts.

This implementation requires the [cnn library](https://github.com/clab/cnn-v1) and you can find pretrained word embeddings for English and Chinese in https://github.com/LeonCrashCode/InOrderParser. 

## Building
The boost version is 1.5.4.

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make

## Experiments

#### Data

You could use the scripts to convert the format of training, development and test data, respectively.

    python ./scripts/get_oracle.py [en|ch] [training data in bracketed format] [training data in bracketed format] > [training oracle]
    python ./scripts/get_oracle.py [en|ch] [training data in bracketed format] [development data in bracketed format] > [development oracle]   
    python ./scripts/get_oracle.py [en|ch] [training data in bracketed format] [test data in bracketed format] > [test oracle]

#### Training

    mkdir model/
    ./build/impl/Kparser --cnn-mem 1700 --training_data [training oracle] --dev_data [development oracle] --bracketing_dev_data [development data in bracketed format] -P -t --pretrained_dim 100 -w [pretrained word embeddings] --lstm_input_dim 128 --hidden_dim 128 -D 0.2

#### Test
    
    ./build/impl/Kparser --cnn-mem 1700 --training_data [training oracle] --test_data [test oracle] --bracketing_dev_data [test data in bracketed format] -P --pretrained_dim 100 -w [pretrained word embeddings] --lstm_input_dim 128 --hidden_dim 128 -m [model file]

The automatically generated file test.eval is the result file.

For more information, please visit https://github.com/LeonCrashCode/InOrderParser.

## Citation
	@article{FERNANDEZGONZALEZ2019559,
		title = "Faster shift-reduce constituent parsing with a non-binary, bottom-up strategy",
		journal = "Artificial Intelligence",
		volume = "275",
		pages = "559 - 574",
		year = "2019",
		issn = "0004-3702",
		doi = "https://doi.org/10.1016/j.artint.2019.07.006",
		url = "http://www.sciencedirect.com/science/article/pii/S000437021830540X",
		author = "Daniel Fernández-González and Carlos Gómez-Rodríguez",
		keywords = "Automata, Natural language processing, Computational linguistics, Parsing, Constituent parsing"
	}

## Acknowledgments

This work has received funding from the European Research Council (ERC), under the European Union's Horizon 2020 research and innovation programme (FASTPARSE, grant agreement No 714150), from MINECO (FFI2014-51978-C2-2-R, TIN2017-85160-C2-1-R) and from Xunta de Galicia (ED431B 2017/01).
