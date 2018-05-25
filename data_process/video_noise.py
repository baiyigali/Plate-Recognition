import os, cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(image):
    cv2.namedWindow("new")
    cv2.imshow("new", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(image, name, folder='./'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, name)
    cv2.imwrite(path, image)
    print("save image at {}".format(path))


# Function to add camera noise
def add_camera_noise(input_irrad_photons, qe=0.69, sensitivity=5.88,
                     dark_noise=2.29, bitdepth=12, baseline=100,
                     rs=np.random.RandomState(seed=42)):
    # Add shot noise
    photons = rs.poisson(input_irrad_photons, size=input_irrad_photons.shape)
    print(np.max(photons), np.min(photons))

    # Convert to electrons
    electrons = qe * photons

    # Add dark noise
    electrons_out = rs.normal(scale=dark_noise, size=electrons.shape) + electrons
    print(np.max(electrons_out), np.min(electrons_out))
    # Convert to ADU and add baseline
    max_adu = np.int(2 ** bitdepth - 1)
    adu = (electrons_out * sensitivity).astype(np.int)  # Convert to discrete numbers
    print(np.max(adu), np.min(adu))
    adu[adu > max_adu] = bitdepth  # models pixel saturation
    adu += baseline

    return adu


image = cv2.imread('./1.png')
image = add_camera_noise(image, sensitivity=5.88, dark_noise=5.)
# print(image)
image = image * 255. / (np.max(image) - np.min(image))
save_image(image, '2.png')
# imshow(image)
plt.imshow(image)
plt.show()
plt.close()
