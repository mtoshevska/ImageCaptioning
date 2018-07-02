from .pipeline import generate_config, get_tokenizer
from .model import create_model
from .generate_roi_features import load_model, generate_features
from .generate_caption import gen_captions


def generate_densecap_captions(image_path):
    config_dict = generate_config(data_dir='data/', mode='debug')
    model = create_model(config_dict=config_dict, compile_model=False)
    model.load_weights('models/weights_3-35.hdf5')
    tokenizer = get_tokenizer(config_dict=config_dict, data_dir='data/')
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}
    model_rcnn = load_model()
    image_embedding = generate_features(image_path, model_rcnn)
    res = gen_captions(config=config_dict, model=model, image_embedding=image_embedding, tokenizer=tokenizer,
                       num_captions=3, index_to_word=index_to_word)
    return res[0][0]
