# SA-Test-Fix

SA-Test-Fix is a Python tool for testing and fixing sentiment analysis models using metamorphic testing and contrastive learning.

## Usage

### Configuration

Edit the `config.ini` file to configure the tool:

```ini
[Paths]
# Path to the sentiment dictionary file
sentiment_dict_path = SiSO/EN-SentiData/EmotionLookupTable.txt

[Model]
# Type of model to use (bert, roberta, etc.)
model_type = bert

[GA]
# Genetic algorithm parameters
population_size = 50
num_generations = 20
crossover_rate = 0.8
mutation_rate = 0.2

[Training]
# Training parameters
total_epochs = 30
rep_phase_ratio = 0.3
joint_phase_ratio = 0.4
initial_lr = 2e-5
final_lr = 5e-6
temperature = 0.1
initial_lambda = 0.1
final_lambda = 0.9
rebuild_interval = 1
batch_size = 16
```

### Python API

```python
from SA_Test_Fix import DataHandler, load_model, GASearch, DefectDetector, MultiStageTrainer, Evaluator

# Load model
model = load_model("path/to/model", "bert")

# Load dataset
data_handler = DataHandler()
dataset = data_handler.load_dataset("sst2", "test")

# Generate test cases
search = GASearch(model, data_handler.tokenizer, mutators)
test_cases = search.generate_test_cases([example['text'] for example in dataset])
search.save_test_cases(test_cases, "output/test_cases.txt")

# Detect defects
detector = DefectDetector(model)
defect_cases, defect_count = detector.detect_defects(data_handler.tokenize_test_cases(test_cases))
detector.save_defect_cases(defect_cases, "output/defect_cases.txt")

# Repair model
trainer = MultiStageTrainer(model)
repaired_model = trainer.train(defect_cases, val_cases)

# Evaluate model
evaluator = Evaluator(repaired_model)
metrics = evaluator.evaluate_model(test_cases, original_model=model)
```

## Project Structure

- `data_handler.py`: Data loading and preprocessing
- `model_wrapper.py`: Model loading and wrapping
- `utils.py`: Utility functions
- `evaluator.py`: Model evaluation
- `metamorphic_mutators.py`: Mutation operators
- `ga_search.py`: Genetic algorithm search framework
- `ga_utils.py`: Genetic algorithm utilities
- `detect.py`: Defect detection
- `sort_score.py`: Sample sorting
- `model_retrainer.py`: Model retraining
- `train.py`: Multi-stage training
- `main.py`: Main entry point
- `config.ini`: Configuration file
- `requirements.txt`: Dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.
