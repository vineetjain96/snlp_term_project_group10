# snlp_term_project_group10

# Encoder Decoder
Instructions to run the code:
  1) python3 reading_pairs.py
  2) python3 preprocessing.py
  3) python3 train.py
  
# Adaptor Grammar
For training:
  python -m launch_train --input_directory=./poetry/ --output_directory=./ --grammar_file=./poetry/grammar.unigram --number_of_documents=18274 --batch_size=10
  
For testing:
  python -m launch_test --input_directory=./poetry/ --model_directory=./poetry/ --non_terminal_symbol=Word
