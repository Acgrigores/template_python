
from pathlib import Path
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from time import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle


def configApp() -> None:
    st.set_page_config(
        page_title="Crunchinator",
        page_icon="🎨",
        layout="centered",
        initial_sidebar_state="auto",
    )

def main() -> None:
    st.title(body="The Crunchinator")
    st.markdown(
        body="Decreases the number of colors in an image",
        help="Created by Anna"
    )
    st.divider()

    st.markdown(body="File Submission")
    st.markdown(body="Submit a .jpg image and pick a number of colors")
    n_colors = st.number_input("Number of colors", min_value=1)
    uploaded_file = st.file_uploader("Your JPG image", type='jpg')


    if uploaded_file is not None:
        # Open the uploaded image
        pic = Image.open(uploaded_file)

        # Display the image
        st.image(pic, caption="Uploaded Image.", use_column_width=True)

        # Convert the image to a NumPy array
        image = np.array(pic)

        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow works well on float data (need to
        # be in the range [0-1])
        array = np.array(image, dtype=np.float64) / 255

        # Load Image and transform to a 2D numpy array.
        w, h, d = original_shape = tuple(image.shape)
        assert d == 3
        image_array = np.reshape(image, (w * h, d))

        t0 = time()
        image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

        # Get labels for all points
        t0 = time()
        labels = kmeans.predict(image_array)


        codebook_random = shuffle(image_array, random_state=0, n_samples=n_colors)
        t0 = time()
        labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)


        def recreate_image(codebook, labels, w, h):
            return codebook[labels].reshape(w, h, -1)


        # Display all results, alongside original image

        plt.figure(3)
        plt.clf()
        plt.axis("off")
        plt.title(f"Quantized image ({n_colors} colors)")
        plt.imshow(recreate_image(codebook_random, labels_random, w, h))
        plt.savefig("crunched.jpg")


        image = Image.open('crunched.jpg')
        st.image(image, caption='Your image crunched down to '+str(n_colors)+" colors")

if __name__ == "__main__":
    configApp()
    main()