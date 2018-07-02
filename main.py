from deeprnn.deeprnn import generate_deeprnn_captions
from densecap.densecap import generate_densecap_captions
import matplotlib.pyplot as plt


def show_caption(caption, image_path):
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption)
    plt.show()


def generate_captions(model_type, image_path):
    assert model_type in ['deeprnn', 'densecap']
    if model_type == 'deeprnn':
        caption = generate_deeprnn_captions(image_path)
    else:
        caption = generate_densecap_captions(image_path)
    show_caption(caption, image_path)


if __name__ == '__main__':
    generate_captions('deeprnn', 'data/visual genome/1.jpg')
    generate_captions('densecap', 'data/flickr8k/72218201_e0e9c7d65b.jpg')
