# Paper: https://arxiv.org/abs/1511.06434v2

import os
import pickle
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


LATENT_DIM = 100
OUTPUT_DIM = 3
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 128
N_EPOCH = 1
FIXED_SEED = 1729

INITIAL_EPOCH = 0
LOSS_G = []
LOSS_D = []
FIXED_NOISE = tf.random.normal([36, 1, 1, LATENT_DIM])

THIS_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(THIS_DIR, 'DCGAN', 'output/')
CHECKPOINT_DIR = os.path.join(THIS_DIR, 'DCGAN', 'checkpoint/')
STATE_FILE_PATH = os.path.join(THIS_DIR, 'DCGAN', 'state_file')


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=FIXED_SEED)
        self.model = keras.Sequential([
            keras.layers.Conv2DTranspose(512, (4, 4), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer=self.w_init, input_shape=(1, 1, LATENT_DIM), data_format='channels_last'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),

            keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=self.w_init),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),

            keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=self.w_init),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),

            keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=self.w_init),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),

            keras.layers.Conv2DTranspose(OUTPUT_DIM, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=self.w_init, activation='tanh'),
        ])

    def call(self, inputs):
        return self.model(inputs)
    

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=FIXED_SEED)
        self.model = keras.Sequential([
            keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.w_init, input_shape=(IMG_HEIGHT, IMG_WIDTH, OUTPUT_DIM), data_format='channels_last'),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.w_init),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.w_init),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.w_init),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='valid', kernel_initializer=self.w_init, activation='sigmoid'),
            keras.layers.Flatten()
        ])

    def call(self, inputs):
        return self.model(inputs)
    

class DCGAN(keras.Model):
    def __init__(self, generator, discriminator):
        super(DCGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, loss_fn, gen_optimizer, disc_optimizer):
        self.loss_fn = loss_fn
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

    @tf.function
    def train_step(self, img_batch):
        batch_size = img_batch.shape[0]
        noise = tf.random.normal([batch_size, 1, 1, LATENT_DIM])

        real_label = np.ones((batch_size, 1))
        fake_label = np.zeros((batch_size, 1))

        with tf.GradientTape() as gen_tape:
            generated_imgs = self.generator(noise)

            with tf.GradientTape() as disc_tape:
                Dx = self.discriminator(img_batch)
                DGz1 = self.discriminator(generated_imgs)
                disc_loss = self.loss_fn(real_label, Dx) + self.loss_fn(fake_label, DGz1)

            disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

            DGz2 = self.discriminator(generated_imgs)
            gen_loss = self.loss_fn(real_label, DGz2)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return gen_loss, disc_loss, Dx, DGz1, DGz2

    def train(self, dataset, n_epoch, fixed_noise):    
        for epoch in range(1, n_epoch+1):
            for i, img_batch in enumerate(dataset):
                gen_loss, disc_loss, Dx, DGz1, DGz2 = self.train_step(img_batch)
                LOSS_G.append(gen_loss.numpy())
                LOSS_D.append(disc_loss.numpy())

                if (i%10 == 0) or (i == len(dataset)-1):
                    Dx = tf.reduce_mean(Dx)
                    DGz1 = tf.reduce_mean(DGz1)
                    DGz2 = tf.reduce_mean(DGz2)
                    print(f'epoch {epoch+INITIAL_EPOCH}/{n_epoch+INITIAL_EPOCH} \t batch {i+1}/{len(dataset)} \t loss_g {gen_loss:.4f} \t loss_d {disc_loss:.4f} \t Dx {Dx:.4f} \t DGz1 {DGz1:.4f} \t DGz2 {DGz2:.4f}')
                    
                    # save image
                    generated_imgs = self.generator(fixed_noise)
                    generated_imgs = (generated_imgs + 1) * 127.5 # convert back to [0, 255]

                    plt.figure(figsize=(10, 10))
                    for a in range(36):
                        plt.subplot(6, 6, a+1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.grid(False)
                        plt.imshow(generated_imgs[a].numpy().astype(np.uint8))
                    plt.savefig(OUTPUT_DIR + f'epoch{epoch+INITIAL_EPOCH}batch{i+1}.png', bbox_inches='tight')
                    plt.close('all')

            # save weight
            self.save_weights(CHECKPOINT_DIR + f'dcgan_epoch{epoch+INITIAL_EPOCH}')

            # save state
            state_file = open(STATE_FILE_PATH, 'wb')
            pickle.dump({
                'LATEST_EPOCH': epoch+INITIAL_EPOCH,
                'LOSS_G': LOSS_G,
                'LOSS_D': LOSS_D,
                'FIXED_NOISE': FIXED_NOISE
            }, state_file)
            state_file.close()

    def generate_image_and_save(self, filename):
        noise = tf.random.normal([1, 1, 1, LATENT_DIM])
        generated_img = self.generator(noise)[0]
        generated_img = (generated_img + 1) * 127.5 # convert back to [0, 255]
        keras.utils.save_img(filename, generated_img, data_format='channels_last')


if __name__ == '__main__':

    # load state
    if os.path.exists(STATE_FILE_PATH):
        print('load state...')
        state_file = open(STATE_FILE_PATH, 'rb')
        state = pickle.load(state_file)
        if 'LATEST_EPOCH' in state:
            INITIAL_EPOCH = state['LATEST_EPOCH']
        if 'LOSS_G' in state:
            LOSS_G = state['LOSS_G']
        if 'LOSS_D' in state:
            LOSS_D = state['LOSS_D']
        if 'FIXED_NOISE' in state:
            FIXED_NOISE = state['FIXED_NOISE']
        state_file.close()

    generator = Generator()
    discriminator = Discriminator()
    dcgan = DCGAN(generator, discriminator)

    latest_cp = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_cp:
        print('load weight...')
        dcgan.load_weights(latest_cp)

    loss_fn = keras.losses.BinaryCrossentropy()
    gen_optimizer = keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)
    disc_optimizer = keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)

    # load dataset & normalize to [-1, 1]
    print('load dataset...')
    dataset = keras.utils.image_dataset_from_directory(
        directory='datasets/waifu',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    ).map(lambda imgs, _ : tf.cast(imgs, tf.float32) / 127.5 - 1)

    dcgan.compile(loss_fn, gen_optimizer, disc_optimizer)
    print('start training!')
    dcgan.train(dataset, N_EPOCH, FIXED_NOISE)
    # dcgan.generate_image_and_save('generated.png')
