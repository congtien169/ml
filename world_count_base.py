import numpy as np

def create_co_matrix(corpus, vocab_size, window_size=1):
    '''create co-occurence matrix
    :param corpus: danh sách word id
    :param vocab_size:số từ
    :param window_size: window size
    :return: co-occurence matrix
    '''
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word

text = u'Nam và gia đình mới mua nhà mới.'
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)
#[0 1 2 3 4 1 5 6]
print(id_to_word)
#{0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}
result = create_co_matrix(corpus, 8, window_size=1)
print(result)