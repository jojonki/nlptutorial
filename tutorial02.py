# http://www.phontron.com/slides/nlp-programming-ja-01-unigramlm.pdf
import math

SOS = '<s>'
EOS = '</s>'


def load_data(fname):
    with open(fname, 'r') as f:
        lines = [[SOS] + l.strip().split(' ' ) + [EOS] for l in f.readlines()]
        flat_lines = [w for l in lines for w in l]
        n_tokens = len(flat_lines) - 2 * len(lines) # remove SOS and EOS counts
        vocabs = sorted(list(set(flat_lines)))
        unigram_freq = {w: flat_lines.count(w) for w in vocabs}

        bigrams = []
        for l in lines:
            for i in range(0, len(l) - 1):
               bigrams.append(' '.join(l[i:i+2]))
        uniq_bigrams = sorted(list(set(bigrams)))
        bigram_freq = {b: bigrams.count(b) for b in uniq_bigrams}

        print('bigrams:', uniq_bigrams)
        print('bigram freq:', bigram_freq)

        return {
            'lines'        : lines,
            'n_tokens'     : n_tokens,
            'vocabs'       : vocabs,
            'uniq_bigrams' : uniq_bigrams,
            'unigram_freq' : unigram_freq,
            'bigram_freq'  : bigram_freq
        }


def train_bigram(data):
    print('-----train_unigram----')
    vocabs = data['vocabs']
    unigram_freq = data['unigram_freq']
    bigram_freq = data['bigram_freq']
    # bigram_tokens = sum(bigram_freq.values())
    # n_tokens = data['n_tokens']

    # print('n_tokens:', n_tokens)
    print('vocabs:', vocabs)
    print('unigram_freq:', unigram_freq)

    print('Training results:----------')
    for big in bigram_freq.keys():
        toks = big.split(' ')
        p_w = bigram_freq[big] / unigram_freq[toks[0]]
        print('P({})={}'.format(big, p_w))


def test_bigram(train_data, test_data):
    print('-----test_unigram----')
    lam2 = 0.3 # bi-gram
    lam1 = 0.3 # uni-gram
    V = 1e6 # 未知語を含む語彙数

    entropy = 0
    log_likelihood = 0
    log2_likelihood = 0
    for l in test_data['lines']:
        p_sent = 1.
        for i in range(0, len(l) - 1):
            w_prev = l[i]
            w = l[i+1]
            bi_token = ' '.join(l[i:i+2])

            if w in train_data['vocabs']:
                p_ml = train_data['unigram_freq'][w] / train_data['n_tokens']
            else:
                p_ml = 0
            p_uni = lam1 * p_ml + (1 - lam1) * (1.0 / V)

            if bi_token in train_data['bigram_freq']:
                p_bi = train_data['bigram_freq'][bi_token] / train_data['unigram_freq'][w_prev]
            else:
                p_bi = 0

            p_bi = lam2 * p_bi + (1 - lam2) * p_uni
            # print('p_uni={}\np_bi={}'.format(p_uni, p_bi))
            # return

            print('P({})={}'.format(bi_token, p_bi))
            p_sent *= p_bi

        log_likelihood += math.log(p_sent)
        log2_likelihood += (-1) * math.log2(p_sent)

    print('Test results:--------')
    print('Log-likelihood:', log_likelihood)
    entropy = log2_likelihood / (test_data['n_tokens'])
    ppl = math.pow(2, entropy)
    print('H={}\nW={}'.format(log2_likelihood, test_data['n_tokens']))
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
    train_data = load_data('data/wiki-en-train.word')
    # train_data = load_data('test/01-train-input.txt')
    # test_data = load_data('test/02-train-input.txt')
    test_data = load_data('data/wiki-en-test.word')

    train_bigram(train_data)
    test_bigram(train_data, test_data)


if __name__ == '__main__':
    main()
