import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib


def data_provider(img_height, img_width, batch_size, split, mpath=''):

    data_dir = pathlib.Path(mpath + 'SmallDataset')
    data_dir_mask = pathlib.Path(mpath + 'SmallDataset/Mask-resized/')
    image_count = len(list(data_dir_mask.glob('*.jpg')))
    print(image_count)

    Mask = list(data_dir_mask.glob('*.jpg'))
    im1 = PIL.Image.open(str(Mask[0]))
    # plt.imshow(im1)
    # plt.show()

    data_dir_nomask = pathlib.Path(mpath + 'SmallDataset/NoMask-resized/')
    image_count = len(list(data_dir_nomask.glob('*.jpg')))
    print(image_count)
    NoMask = list(data_dir_nomask.glob('*.jpg'))
    im2 = PIL.Image.open(str(NoMask[0]))

    data_dir_oth = pathlib.Path(mpath + 'SmallDataset/NotPerson-resized/')
    image_count = len(list(data_dir_oth.glob('*.jpg')))
    print(image_count)

    Oth = list(data_dir_oth.glob('*.jpg'))
    im3 = PIL.Image.open(str(Oth[0]))

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(im1)
    ax[1].imshow(im2)
    ax[2].imshow(im3)

    plt.show()

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=split,
        subset='training',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=split,
        subset='validation',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    print(len(train_ds), len(val_ds))

    return train_ds, val_ds



# data_provider(200, 200, 32, 0.2)

