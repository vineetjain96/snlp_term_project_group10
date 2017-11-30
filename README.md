# snlp_term_project_group10

# Encoder Decoder
  python3 reading_pairs.py
  python3 preprocessing.py
  python3 train.py
  
# Adaptor Grammar
For training:
  python -m launch_train --input_directory=./poetry/ --output_directory=./ --grammar_file=./poetry/grammar.unigram --number_of_documents=18274 --batch_size=10
  
For testing:
  python -m launch_test --input_directory=./poetry/ --model_directory=./poetry/ --non_terminal_symbol=Word
