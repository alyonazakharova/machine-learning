import imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

pic = np.array(imageio.imread('data/pic.jpg'))

data = pic / 255.0
data = data.reshape(pic.shape[0] * pic.shape[1], 3)

model = KMeans(n_clusters=8)
model.fit(data)
reduced_colors = model.cluster_centers_[model.predict(data)]

new_pic = reduced_colors.reshape(pic.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(pic)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(new_pic)
ax[1].set_title('8-color Image', size=16)
plt.show()
