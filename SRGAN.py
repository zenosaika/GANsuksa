# Paper: https://arxiv.org/abs/1609.04802v5

import os
import pickle
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


LATENT_DIM = 100
OUTPUT_DIM = 3
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32
N_EPOCH = 1
FIXED_SEED = 1729

INITIAL_EPOCH = 0
LOSS_G = []
LOSS_D = []
FIXED_IMG = None

THIS_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(THIS_DIR, 'SRGAN', 'output/')
CHECKPOINT_DIR = os.path.join(THIS_DIR, 'SRGAN', 'checkpoint/')
STATE_FILE_PATH = os.path.join(THIS_DIR, 'SRGAN', 'state_file')


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=FIXED_SEED)
        self.model = self.make_generator()

    def make_generator(self):
        inputs = keras.Input(shape=(None, None, 3))
        x = keras.layers.Conv2D(64, (9, 9), strides=(1, 1), padding='same', use_bias=False)(inputs)
        x = keras.layers.PReLU(shared_axes=[1, 2])(x)
        tmp = x

        for _ in range(8):
            x = self.residual_block(x)

        x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Add()([x, tmp])

        for _ in range(2):
            x = self.upsample_block(x)

        outputs = keras.layers.Conv2D(3, (9, 9), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(x)
        return keras.Model(inputs, outputs)
        
    def residual_block(self, inputs):
        x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Add()([x, inputs])
        return outputs

    def upsample_block(self, inputs):
        x = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
        x = tf.nn.depth_to_space(x, 2)
        outputs = keras.layers.PReLU(shared_axes=[1, 2])(x)
        return outputs

    def call(self, inputs):
        return self.model(inputs)
    

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = keras.Sequential()

        self.model.add(keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, OUTPUT_DIM), data_format='channels_last'))
        self.model.add(keras.layers.LeakyReLU(0.2))

        n_filters = [64, 128, 128, 256, 256, 512, 512]
        n_strides = [2, 1, 2, 1, 2, 1, 2]
        for i in range(7):
            self.model.add(keras.layers.Conv2D(n_filters[i], (3, 3), strides=(n_strides[i], n_strides[i]), padding='same', use_bias=False))
            self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.LeakyReLU(0.2))

        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(1024))
        self.model.add(keras.layers.LeakyReLU(0.2))
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))

    def call(self, inputs):
        return self.model(inputs)
    

class SRGAN(keras.Model):
    def __init__(self, generator, discriminator):
        super(SRGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.vgg19 = keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        self.vgg19.trainable = False
        
        for l in self.vgg19.layers:
            l.trainable = False

        self.features_extractor = keras.Model(inputs=self.vgg19.input, outputs=self.vgg19.get_layer('block5_conv4').output)

    def compile(self, loss_fn, gen_optimizer, disc_optimizer):
        self.loss_fn = loss_fn
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

    def preproces_vgg(self, img):
        # convert back to [0, 255]
        img = (img + 1) * 127.5
        
        # RGB to BGR
        img = img[..., ::-1]
        
        # apply Imagenet preprocessing : BGR mean
        mean = [103.939, 116.778, 123.68]
        IMAGENET_MEAN = tf.constant(-np.array(mean))
        img = keras.backend.bias_add(img, keras.backend.cast(IMAGENET_MEAN, keras.backend.dtype(img)))
    
        return img

    def vgg_loss(self, hr, sr):
        features_sr = self.features_extractor(self.preproces_vgg(sr))
        features_hr = self.features_extractor(self.preproces_vgg(hr))
        
        return 0.006 * keras.backend.mean(keras.backend.square(features_sr - features_hr), axis=-1)
        
    @tf.function
    def train_step(self, img_batch):
        batch_size = img_batch.shape[0]

        lr = tf.image.resize(img_batch, [64, 64])

        real_label = np.ones((batch_size, 1))
        fake_label = np.zeros((batch_size, 1))

        with tf.GradientTape() as gen_tape:
            sr = self.generator(lr, training=True)

            with tf.GradientTape() as disc_tape:
                Dhr = self.discriminator(img_batch, training=True)
                Dsr1 = self.discriminator(sr, training=True)
                disc_loss = self.loss_fn(real_label, Dhr) + self.loss_fn(fake_label, Dsr1)

            disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

            Dsr2 = self.discriminator(sr, training=True)
            content_loss = self.vgg_loss(img_batch, sr)
            gen_loss = tf.reduce_mean(content_loss) + 1e-3 * self.loss_fn(real_label, Dsr2)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return gen_loss, disc_loss, Dhr, Dsr1, Dsr2, content_loss

    def train(self, dataset, n_epoch, fixed_img):    
        for epoch in range(1, n_epoch+1):
            for i, img_batch in enumerate(dataset):
                gen_loss, disc_loss, Dhr, Dsr1, Dsr2, content_loss = self.train_step(img_batch)
                LOSS_G.append(gen_loss.numpy())
                LOSS_D.append(disc_loss.numpy())

                if (i%10 == 0) or (i == len(dataset)-1):
                    Dhr = tf.reduce_mean(Dhr)
                    Dsr1 = tf.reduce_mean(Dsr1)
                    Dsr2 = tf.reduce_mean(Dsr2)
                    content_loss = tf.reduce_mean(content_loss)
                    print(f'epoch {epoch+INITIAL_EPOCH}/{n_epoch+INITIAL_EPOCH} \t batch {i+1}/{len(dataset)} \t loss_g {gen_loss:.4f} \t loss_d {disc_loss:.4f} \t Dhr {Dhr:.4f} \t Dsr1 {Dsr1:.4f} \t Dsr2 {Dsr2:.4f} \t vgg_loss {content_loss:.4f}')
                    
                    # save image
                    sr_img = self.generator(np.array([fixed_img]), training=False)[0]
                    sr_img = (sr_img + 1) * 127.5 # convert back to [0, 255]
                    keras.utils.save_img(OUTPUT_DIR + f'epoch{epoch+INITIAL_EPOCH}batch{i+1}.png', sr_img, data_format='channels_last')

            # save weight
            self.save_weights(CHECKPOINT_DIR + f'srgan_epoch{epoch+INITIAL_EPOCH}')

            # save state
            state_file = open(STATE_FILE_PATH, 'wb')
            pickle.dump({
                'LATEST_EPOCH': epoch+INITIAL_EPOCH,
                'LOSS_G': LOSS_G,
                'LOSS_D': LOSS_D,
                'FIXED_IMG': FIXED_IMG
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
        if 'LOSS_D' in state:
            LOSS_D = state['LOSS_D']
        if 'FIXED_IMG' in state:
            FIXED_IMG = state['FIXED_IMG']
        state_file.close()

    generator = Generator()
    discriminator = Discriminator()
    srgan = SRGAN(generator, discriminator)

    latest_cp = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_cp:
        print('load weight...')
        srgan.load_weights(latest_cp)

    loss_fn = keras.losses.BinaryCrossentropy()
    gen_optimizer = keras.optimizers.legacy.Adam(1e-4, beta_1=0.9)
    disc_optimizer = keras.optimizers.legacy.Adam(1e-4, beta_1=0.9)

    # load dataset & normalize to [-1, 1]
    print('load dataset...')
    dataset = keras.utils.image_dataset_from_directory(
        directory='datasets/hr_waifu',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    ).map(lambda imgs, _ : tf.cast(imgs, tf.float32) / 127.5 - 1)

    if FIXED_IMG is None:
        img = keras.utils.load_img('SRGAN/fixed_img.jpg', target_size=(64, 64))
        img = keras.utils.img_to_array(img) / 127.5 - 1
        FIXED_IMG = img

    srgan.compile(loss_fn, gen_optimizer, disc_optimizer)

    print('start training!')
    srgan.train(dataset, N_EPOCH, FIXED_IMG)

    # tf.keras.utils.plot_model(generator.model, to_file='generator.png', show_shapes=True)
    # tf.keras.utils.plot_model(discriminator.model, to_file='discriminator.png', show_shapes=True)
