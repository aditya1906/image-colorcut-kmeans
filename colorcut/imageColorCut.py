import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import MiniBatchKMeans

def imageColorCut(imagePath, k, newImagePath=None):
    """
    imageColorCut is the main method
    that create a color cut effect on the image.
    Parameters:
      imagePath(type: str)(Required) : Path of the image.
      k(type: int)(Requied) : The number of colors to be present.
      newImagePath(type: str)(Optional) : Path of the new image, also be used to name the newly generated image.
        default: (name-of-the-image)-(k).jpg
    """
    if type(imagePath).__name__ != 'str':
        raise ValueError("illegal imagePath provided. imagePath should be a string")
    if type(k).__name__ != 'int':
        raise ValueError("illegal k provided. k should be an integer")
    img = io.imread(imagePath)
    img_data = (img / 255.0).reshape(-1, 3)

    kmeans = MiniBatchKMeans(k).fit(img_data)
    k_colors = kmeans.cluster_centers_[kmeans.predict(img_data)]
    k_img = np.reshape(k_colors, (img.shape))
    
    if type(newImagePath).__name__ == 'str':
        plt.imsave(newImagePath, k_img)
    else:
        imageName = imagePath.split('.')
        imageName = '{0}-{1}.{2}'.format('.'.join(imageName[:-1]), k, imageName[-1])
        plt.imsave(imageName, k_img)
    print('Successfully generated your image')
