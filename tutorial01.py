# http://www.phontron.com/slides/nlp-programming-ja-01-unigramlm.pdf
import math

EOS = '</s>'

def load_data(fname):
    with open(fname, 'r') as f:
        lines = [l.strip().split(' ' ) + [EOS] for l in f.readlines()]
        flat_lines = [w for l in lines for w in l]
        n_tokens = len(flat_lines)
        vocabs = sorted(list(set(flat_lines)))
        word_freq = {w: flat_lines.count(w) for w in vocabs}

        return {
            'lines'     : lines,
            'n_tokens'  : n_tokens,
            'vocabs'    : vocabs,
            'word_freq' : word_freq
        }


def train_unigram(data):
    print('-----train_unigram----')
    vocabs = data['vocabs']
    word_freq = data['word_freq']
    n_tokens = data['n_tokens']

    print('n_tokens:', n_tokens)
    print('vocabs:', vocabs)
    print('word_freq:', word_freq)

    print('Training results:----------')
    for w in vocabs:
        p_w = word_freq[w] / n_tokens
        print('P({})={}'.format(w, p_w))


def test_unigram(train_data, test_data):
    print('-----test_unigram----')
    lam = 0.95
    N = 1000000 # 未知語を含む語彙数

    entropy = 0
    log_likelihood = 0
    log2_likelihood = 0
    for l in test_data['lines']:
        p_sent = 1.0
        for w in l:
            if w in train_data['vocabs']:
                p_ml = train_data['word_freq'][w] / train_data['n_tokens']
            else:
                p_ml = 0
            p_w = lam * p_ml + (1 - lam) * (1.0 / N)
            print('P({})={}'.format(w, p_w))
            p_sent *= p_w
        log_likelihood += math.log(p_sent)
        log2_likelihood += (-1) * math.log(p_sent, 2)

    print('Test results:--------')
    print('Log-likelihood:', log_likelihood)
    entropy = log2_likelihood / (test_data['n_tokens']) # ignore </s> token
    ppl = math.pow(2, entropy)
    print('entropy:', entropy)
    print('perplexity:', ppl)

    # calculate coverage
    test_flat_lines = [w for l in test_data['lines'] for w in l]
    contains_count = 0
    for w in test_flat_lines:
        if w in train_data['vocabs']:
            contains_count += 1
    print('coverage:', contains_count / len(test_flat_lines))


def main():
    # train_data = load_data('test/01-train-input.txt')
    # test_data = load_data('test/01-test-input.txt')
    train_data = load_data('data/wiki-en-train.word')
    test_data = load_data('data/wiki-en-test.word')

    train_unigram(train_data)
    test_unigram(train_data, test_data)


if __name__ == '__main__':
    main()
