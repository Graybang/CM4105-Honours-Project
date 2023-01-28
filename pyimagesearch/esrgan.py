# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow.nn import depth_to_space
from tensorflow.keras import Model
from tensorflow.keras import Input

class ESRGAN(object):
	@staticmethod
	def generator(scalingFactor, featureMaps, residualBlocks,
			leakyAlpha, residualScalar):
		# initialize the input layer
		inputs = Input((None, None, 3))
		xIn = Rescaling(scale=1.0/255, offset=0.0)(inputs)

		# pass the input through CONV => LeakyReLU block
		xIn = Conv2D(featureMaps, 9, padding="same")(xIn)
		xIn = LeakyReLU(leakyAlpha)(xIn)

        # construct the residual in residual block
		x = Conv2D(featureMaps, 3, padding="same")(xIn)
		x1 = LeakyReLU(leakyAlpha)(x)
		x1 = Add()([xIn, x1])
		x = Conv2D(featureMaps, 3, padding="same")(x1)
		x2 = LeakyReLU(leakyAlpha)(x)
		x2 = Add()([x1, x2])
		x = Conv2D(featureMaps, 3, padding="same")(x2)
		x3 = LeakyReLU(leakyAlpha)(x)
		x3 = Add()([x2, x3])
		x = Conv2D(featureMaps, 3, padding="same")(x3)
		x4 = LeakyReLU(leakyAlpha)(x)
		x4 = Add()([x3, x4])
		x4 = Conv2D(featureMaps, 3, padding="same")(x4)
		xSkip = Add()([xIn, x4])

		# scale the residual outputs with a scalar between [0,1]
		xSkip = Lambda(lambda x: x * residualScalar)(xSkip)

        # create a number of residual in residual blocks
		for blockId in range(residualBlocks-1):
			x = Conv2D(featureMaps, 3, padding="same")(xSkip)
			x1 = LeakyReLU(leakyAlpha)(x)
			x1 = Add()([xSkip, x1])
			x = Conv2D(featureMaps, 3, padding="same")(x1)
			x2 = LeakyReLU(leakyAlpha)(x)
			x2 = Add()([x1, x2])
			x = Conv2D(featureMaps, 3, padding="same")(x2)
			x3 = LeakyReLU(leakyAlpha)(x)
			x3 = Add()([x2, x3])
			x = Conv2D(featureMaps, 3, padding="same")(x3)
			x4 = LeakyReLU(leakyAlpha)(x)
			x4 = Add()([x3, x4])
			x4 = Conv2D(featureMaps, 3, padding="same")(x4)
			xSkip = Add()([xSkip, x4])
			xSkip = Lambda(lambda x: x * residualScalar)(xSkip)

        # process the residual output with a conv kernel
		x = Conv2D(featureMaps, 3, padding="same")(xSkip)
		x = Add()([xIn, x])
		
        # upscale the image with pixel shuffle
		x = Conv2D(featureMaps * (scalingFactor // 2), 3,
			padding="same")(x)
		x = tf.nn.depth_to_space(x, 2)
		x = LeakyReLU(leakyAlpha)(x)
		
        # upscale the image with pixel shuffle
		x = Conv2D(featureMaps, 3, padding="same")(x)
		x = tf.nn.depth_to_space(x, 2)
		x = LeakyReLU(leakyAlpha)(x)
		
        # get the output layer
		x = Conv2D(3, 9, padding="same", activation="tanh")(x)
		output = Rescaling(scale=127.5, offset=127.5)(x)
		
        # create the generator model
		generator = Model(inputs, output)
		
        # return the generator model
		return generator

	@staticmethod
	def discriminator(featureMaps, leakyAlpha, discBlocks):
		# initialize the input layer and process it with conv kernel
		inputs = Input((None, None, 3))
		x = Rescaling(scale=1.0/127.5, offset=-1)(inputs)
		x = Conv2D(featureMaps, 3, padding="same")(x)
		x = LeakyReLU(leakyAlpha)(x)
		
		# pass the output from previous layer through a CONV => BN =>
		# LeakyReLU block
		x = Conv2D(featureMaps, 3, padding="same")(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(leakyAlpha)(x)