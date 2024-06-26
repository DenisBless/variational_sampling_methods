from flax import linen as nn  # The Linen API


class CNN(nn.Module):

    @nn.compact
    # Provide a constructor to register a new parameter
    # and return its initial value
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)  # There are 10 classes in MNIST
        return x


class FashionMnistCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        x = nn.Dense(features=128)(x)
        x = nn.relu(x)

        x = nn.Dense(features=10)(x)  # There are 10 classes in MNIST
        return x