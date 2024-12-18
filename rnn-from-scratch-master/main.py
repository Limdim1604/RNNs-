import numpy as np
import random
from rnn import RNN
from data import train_sentences

train_sentences = [sentence.lower() for sentence in train_sentences]

# Tạo từ điển từ dữ liệu trong file data.py
vocab = list(set(word for sentence in train_sentences for word in sentence.split()))
vocab.append("<UNK>")  # Thêm token đặc biệt
vocab_size = len(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}


# Tạo dữ liệu huấn luyện (đầu vào và từ tiếp theo)
def create_training_data(sentences):
    data = []
    for sentence in sentences:
        words = sentence.split()
        for start in range(len(words)):  # Tạo tất cả các chuỗi con
            for end in range(start + 1, len(words)):
                input_seq = words[start:end]
                target_word = words[end]
                data.append((input_seq, target_word))
    return data


training_data = create_training_data(train_sentences)

# Chuyển input thành vector one-hot
def create_inputs(words):
    inputs = []
    for word in words:
        if word not in word_to_idx:
            word = "<UNK>"  # Thay từ chưa biết bằng <UNK>
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[word]] = 1
        inputs.append(v)
    return inputs


def softmax(xs):
    return np.exp(xs) / np.sum(np.exp(xs), axis=0)

# Khởi tạo RNN
rnn = RNN(vocab_size, vocab_size)

# Huấn luyện
def train_rnn(data, epochs=1000):
    for epoch in range(epochs):
        loss = 0
        for input_words, target_word in data:
            inputs = create_inputs(input_words)
            target_idx = word_to_idx[target_word]

            # Forward
            out, _ = rnn.forward(inputs)
            probs = softmax(out)

            # Loss
            loss -= np.log(probs[target_idx])

            # Backprop
            d_L_d_y = probs
            d_L_d_y[target_idx] -= 1
            rnn.backprop(d_L_d_y)

        # Đảm bảo loss là một scalar
        print(f"Epoch {epoch + 1}, Loss: {(loss / len(data)).item():.4f}")

train_rnn(training_data)

# Dự đoán từ tiếp theo
def predict_next_words(input_words, top_k=2):
    inputs = create_inputs(input_words)
    out, _ = rnn.forward(inputs)
    probs = softmax(out)

    # Lấy top_k từ có xác suất cao nhất
    top_indices = np.argsort(probs, axis=0)[-top_k:][::-1]
    predictions = [idx_to_word[idx.item()] for idx in top_indices]
    return predictions


# Vòng lặp nhập liệu
while True:
    # Nhập câu từ người dùng
    user_input = input("Enter a sentence (or type 'exit' to quit): ")
    
    # Kiểm tra nếu người dùng muốn thoát
    if user_input.lower() == 'exit':
        print("Exiting the program. Goodbye!")
        break
    
    # Xử lý câu nhập
    input_words = user_input.lower().split()  # Chuyển sang chữ thường
    
    # Gợi ý từ tiếp theo
    try:
        predicted_words = predict_next_words(input_words, top_k=2)
        print(f"Input: {' '.join(input_words)}")
        print(f"Predicted next words: {', '.join(predicted_words)}")
    except KeyError as e:
        print(f"Error: {e}. Please ensure your input contains words from the vocabulary.")
