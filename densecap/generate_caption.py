import numpy as np
from keras.preprocessing.sequence import pad_sequences


def make_caption_human_readable(caption, index_to_word):
    '''Method to convert the caption vector into human readable format'''
    return (' '.join(
        list(
            map(lambda x: index_to_word[x], caption)
        )
    ))


def gen_captions(config,
                 model,
                 image_embedding,
                 tokenizer,
                 num_captions,
                 index_to_word):
    '''Method to generate the captions given a model, embeddings of the input image
    and a beam size'''
    START = "__START__"
    captions = tokenizer.texts_to_sequences(texts=[START])
    scores = [0.0]
    while (len(captions[0]) < config["max_caption_length"]):
        new_captions = []
        new_scores = []
        for caption, score in zip(captions, scores):
            caption_so_far = pad_sequences(sequences=[caption],
                                           maxlen=config["max_caption_length"],
                                           padding="pre")
            next_word_scores = model.predict(
                [np.reshape(image_embedding, (1, -1)), np.asarray(caption_so_far)]
            )[0]
            candidate_next_words = np.argsort(next_word_scores)[-num_captions:]
            for next_word in candidate_next_words:
                caption_so_far, caption_score_so_far = caption[:], score
                caption_so_far.append(next_word)
                new_captions.append(caption_so_far)
                new_score = score + next_word_scores[next_word]
                new_scores.append(new_score)
        captions = new_captions
        scores = new_scores
        captions_scores_list = list(
            zip(captions, scores))
        captions_scores_list.sort(key=lambda x: x[1])
        captions_scores_list = captions_scores_list[-num_captions:]
        captions, scores = zip(*captions_scores_list)
    results = []
    for (caption, score) in captions_scores_list:
        results.append((make_caption_human_readable(caption=caption, index_to_word=index_to_word), score))
    return results