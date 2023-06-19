import numpy as np
import tensorflow as tf
import pickle


class ThreeCharacterClassicInference:
    def __init__(self, model_path, dictionary_path):
        self.load_model(model_path)
        self.load_dictionary(dictionary_path)

    def load_model(self, model_path):
        if not isinstance(model_path, str):
            raise ValueError("Model path should be a string.")

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def load_dictionary(self, dictionary_path):
        if not isinstance(dictionary_path, str):
            raise ValueError("Dictionary path should be a string.")

        with open(dictionary_path, "rb") as f:
            self.dictionary = pickle.load(f)

    def texts_to_sequence(self, text):
        if not isinstance(text, str):
            raise ValueError("Input text should be a string.")

        return [self.dictionary.get(word, 0) for word in text]

    def sequence_to_text(self, seq):
        if not isinstance(seq, (list, np.ndarray)):
            raise ValueError("Input sequence should be a list or numpy array.")

        inverse_dictionary = {value: key for key, value in self.dictionary.items()}
        return "".join([inverse_dictionary.get(token, "") for token in seq])

    def predict_next_3(self, input_triplets):
        if not isinstance(input_triplets, str) or len(input_triplets) != 3:
            raise ValueError(
                "Input triplets should be a string of exactly 3 characters."
            )

        seq = np.array(self.texts_to_sequence(input_triplets)).reshape(1, -1)
        seq = seq.astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]["index"], seq)
        self.interpreter.invoke()

        result = [
            self.interpreter.get_tensor(self.output_details[i]["index"])
            for i in range(3)
        ]
        tokens = [np.argmax(prob) for prob in result]
        return self.sequence_to_text(tokens)
    def get_dictionary(self):
        return self.dictionary


if __name__ == "__main__":
    model_path = "3character.tflite"
    dictionary_path = "3character_dict.pickle"

    inference = ThreeCharacterClassicInference(model_path, dictionary_path)
    input_triplets = "性相近"
    predicted_triplets = inference.predict_next_3(input_triplets)
    print(predicted_triplets)
