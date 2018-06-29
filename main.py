from deeprnn.deeprnn import generate_deeprnn_captions
import matplotlib.pyplot as plt


def show_caption(caption, image_path):
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption)
    plt.show()


def generate_captions(model_type, image_path):
    assert model_type in ['deeprnn', 'densecaption']
    if model_type == 'deeprnn':
        caption = generate_deeprnn_captions(image_path)
        show_caption(caption, image_path)


if __name__ == '__main__':
    generate_captions('deeprnn', 'data/visual genome/1.jpg')
