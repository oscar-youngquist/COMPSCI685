from abc import ABC, abstractmethod

class Model(ABC):

    def __init__(self, data_path, shared_docs_path, num_examples):
        self.data_path = data_path
        self.shared_docs_path = shared_docs_path
        self.num_examples = num_examples
        
    @abstractmethod
    def get_predicted_summary(self, target_doc, example_summaires, processed_ctr):
        pass