# Paper: https://arxiv.org/abs/1704.00028v3

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
N_CRITIC = 5
LAMBDA = 10
FIXED_SEED = 1729

INITIAL_EPOCH = 0
LOSS_G = []
LOSS_C = []
FIXED_NOISE = tf.random.normal([36, 1, 1, LATENT_DIM])

THIS_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(THIS_DIR, 'WGAN-GP', 'output/')
CHECKPOINT_DIR = os.path.join(THIS_DIR, 'WGAN-GP', 'checkpoint/')
STATE_FILE_PATH = os.path.join(THIS_DIR, 'WGAN-GP', 'state_file')


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


class Critic(keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=FIXED_SEED)
        self.model = keras.Sequential([
            keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.w_init, input_shape=(IMG_HEIGHT, IMG_WIDTH, OUTPUT_DIM), data_format='channels_last'),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.w_init),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.w_init),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.w_init),
            keras.layers.LeakyReLU(0.2),

            keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='valid', kernel_initializer=self.w_init, activation='linear'),
            keras.layers.Flatten(),
        ])

    def call(self, inputs):
        return self.model(inputs)
    

class WGAN_GP(keras.Model):
    def __init__(self, generator, critic):
        super(WGAN_GP, self).__init__()
        self.generator = generator
        self.critic = critic

    def compile(self, gen_optimizer, crit_optimizer):
        self.gen_optimizer = gen_optimizer
        self.crit_optimizer = crit_optimizer

    @tf.function
    def train_step(self, img_batchs):
        # train critic n_critic times
        for img_batch in img_batchs:
            batch_size = img_batch.shape[0]
            noise = tf.random.normal([batch_size, 1, 1, LATENT_DIM])
            epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)

            with tf.GradientTape() as crit_tape:
                with tf.GradientTape() as gp_tape:
                    generated_imgs = self.generator(noise, training=True)
                    interpolated_imgs = epsilon * img_batch + (1-epsilon) * generated_imgs
                    interpolated_imgs_pred = self.critic(interpolated_imgs, training=True)

                grads = gp_tape.gradient(interpolated_imgs_pred, interpolated_imgs)
                grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
                gradient_panelty = tf.reduce_mean(tf.square(grad_norms - 1))

                fx = self.critic(img_batch, training=True)
                fgz1 = self.critic(generated_imgs, training=True)
                crit_loss = tf.reduce_mean(fgz1) - tf.reduce_mean(fx) + LAMBDA * gradient_panelty

            crit_gradients = crit_tape.gradient(crit_loss, self.critic.trainable_variables)
            self.crit_optimizer.apply_gradients(zip(crit_gradients, self.critic.trainable_variables))

        # train generator 1 time
        noise = tf.random.normal([BATCH_SIZE, 1, 1, LATENT_DIM])
        with tf.GradientTape() as gen_tape:
            generated_imgs = self.generator(noise, training=True)
            fgz2 = self.critic(generated_imgs, training=True)
            gen_loss = -tf.reduce_mean(fgz2)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return gen_loss, crit_loss, fx, fgz1, fgz2

    def train(self, dataset, n_epoch, n_critic, fixed_noise):
        dataset = list(dataset)
        n_batch = len(dataset)
        n_iter = n_batch // n_critic

        for epoch in range(1, n_epoch+1):
            iter = 0
            for i in range(0, n_batch, n_critic):
                img_batchs = dataset[i:i+n_critic]
                gen_loss, crit_loss, fx, fgz1, fgz2 = self.train_step(img_batchs)
                LOSS_G.append(gen_loss.numpy())
                LOSS_C.append(crit_loss.numpy())

                if (iter%5 == 0) or (iter == n_iter-1):
                    fx = tf.reduce_mean(fx)
                    fgz1 = tf.reduce_mean(fgz1)
                    fgz2 = tf.reduce_mean(fgz2)
                    print(f'epoch {epoch+INITIAL_EPOCH}/{n_epoch+INITIAL_EPOCH} \t iteration {iter+1}/{n_iter} \t loss_g {gen_loss:.4f} \t loss_c {crit_loss:.4f} \t fx {fx:.4f} \t fgz1 {fgz1:.4f} \t fgz2 {fgz2:.4f}')
                    
                    # save image
                    generated_imgs = self.generator(fixed_noise, training=False)
                    generated_imgs = (generated_imgs + 1) * 127.5 # convert back to [0, 255]

                    plt.figure(figsize=(10, 10))
                    for a in range(36):
                        plt.subplot(6, 6, a+1)
                        plt.xticks([])
                        plt.yticks([])
                        plt.grid(False)
                        plt.imshow(generated_imgs[a].numpy().astype(np.uint8))
                    plt.savefig(OUTPUT_DIR + f'epoch{epoch+INITIAL_EPOCH}iteration{iter+1}.png', bbox_inches='tight')
                    plt.close('all')

                iter += 1

            # save weight
            self.save_weights(CHECKPOINT_DIR + f'wgan-gp_epoch{epoch+INITIAL_EPOCH}')

            # save state
            state_file = open(STATE_FILE_PATH, 'wb')
            pickle.dump({
                'LATEST_EPOCH': epoch+INITIAL_EPOCH,
                'LOSS_G': LOSS_G,
                'LOSS_C': LOSS_C,
                'FIXED_NOISE': FIXED_NOISE
            }, state_file)
            state_file.close()

    def generate_image_and_save(self, filename):
        noise = tf.random.normal([1, 1, 1, LATENT_DIM])
        generated_img = self.generator(noise, training=False)[0]
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
        if 'LOSS_C' in state:
            LOSS_C = state['LOSS_C']
        if 'FIXED_NOISE' in state:
            FIXED_NOISE = state['FIXED_NOISE']
        state_file.close()

    generator = Generator()
    critic = Critic()
    wgan_gp = WGAN_GP(generator, critic)

    latest_cp = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_cp:
        print('load weight...')
        wgan_gp.load_weights(latest_cp)

    gen_optimizer = keras.optimizers.legacy.Adam(1e-4, beta_1=0.0, beta_2=0.9)
    crit_optimizer = keras.optimizers.legacy.Adam(1e-4, beta_1=0.0, beta_2=0.9)

    # load dataset & normalize to [-1, 1]
    print('load dataset...')
    dataset = keras.utils.image_dataset_from_directory(
        directory='datasets/waifu',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    ).map(lambda imgs, _ : tf.cast(imgs, tf.float32) / 127.5 - 1)

    wgan_gp.compile(gen_optimizer, crit_optimizer)
    print('start training!')
    wgan_gp.train(dataset, N_EPOCH, N_CRITIC, FIXED_NOISE)
    # wgan_gp.generate_image_and_save('generated.png')
