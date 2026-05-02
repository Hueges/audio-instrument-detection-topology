import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from gtda.plotting import plot_point_cloud
from gtda.homology import VietorisRipsPersistence
from ripser import ripser
from persim import plot_diagrams
from gtda.diagrams import PersistenceEntropy, PersistenceLandscape
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from gtda.time_series import SingleTakensEmbedding
from gtda.plotting import plot_diagram
from tqdm import tqdm
from gtda.diagrams import HeatKernel

#ucitavamo podatke iz foldera muzika 12 koji sadrzi wav fajl flaute,klavira, trube i violine
B=[]
path2='muzika12'

for filename in os.listdir(path2):
    if filename.split('.')[-1] == 'wav':
        print(filename)
        sampling_rate2, audio_data2 = wav.read(os.path.join(path2, filename))

        # Extract a single channel if stereo audio
        if audio_data2.ndim > 1:
            audio_data2 = audio_data2[:, 0]  # Extract the first channel

        B.append(audio_data2)



#plot nase  vremenske serije

for i in B:
    plt.plot(i)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Reprezentacija note ')
    plt.grid()
    plt.show(block=True)
    plt.close()


# Calculate the indices corresponding to the desired time window
# start_time = 1.0  # Start time in seconds
# end_time = 1.5  # End time in seconds
#
# # Calculate the corresponding indices based on the sampling rate
# start_index = int(start_time * sampling_rate2)
# end_index = int(end_time * sampling_rate2)
#
# # Extract the desired window from the audio data
# window = B[0][start_index:end_index]

# Plot the sound wave of the desired window
# plt.plot(window)
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title('Zvuk note A3-forte na violini (od 1s do 1.5s)')
# plt.grid()
# plt.show(block=True)
# plt.close()


#Pravimo optimalan oblak tacaka
max_embedding_dimension = 100
max_time_delay = 100
stride =20


embedder_periodic2 = SingleTakensEmbedding(
    parameters_type="search",
    time_delay=max_time_delay,
    dimension=max_embedding_dimension,
    stride=stride,
)



def fit_embedder2(embedder: SingleTakensEmbedding, y: np.ndarray, verbose: bool=True) -> np.ndarray:
    """Fits a Takens embedder and displays optimal search parameters."""
    y_embedded = embedder.fit_transform(y)

    return y_embedded



def fit_embedder(embedder: SingleTakensEmbedding, y: np.ndarray, verbose: bool=True) -> np.ndarray:
    """Fits a Takens embedder and displays optimal search parameters."""
    y_embedded = embedder.fit_transform(y)

    if verbose:
        print(f"Shape of embedded time series: {y_embedded.shape}")
        print(
            f"Optimal embedding dimension is {embedder.dimension_} and time delay is {embedder.time_delay_}"
        )
    if embedder.dimension_ == 2:
            # Perform embedding in 3 dimensions instead
            embedding_dimension_periodic = 3
            embedding_time_delay_periodic = embedder.time_delay
            stride = 200

            embedder_periodic = SingleTakensEmbedding(
                parameters_type="fixed",
                n_jobs=2,
                time_delay=embedding_time_delay_periodic,
                dimension=embedding_dimension_periodic,
                stride=stride,
            )

            y_embedded = fit_embedder2(embedder_periodic, y)
            print(f"Updated embedding dimension to 3. New shape: {y_embedded.shape}")


    return y_embedded

d=[]
b=[]

#svaku vremensku seriju pretvaramo u oblak tacaka
for i, audio_data2 in tqdm(enumerate(B)):

    result = fit_embedder(embedder_periodic2, audio_data2)
    d.append(result)


#pomocu PCA svodimo na 3 dimenzije, a ukoliko je optimalna dimenzijaj 2, pozivamo opet fitembeder da napravi 3 dimenzije
for i, arr in tqdm(enumerate(d)):
    if arr.shape[1] > 3:
        print('here', arr.shape)
        pca = PCA(n_components=3)
        d_pca = pca.fit_transform(arr)
        b.append(d_pca)
    else:
        print('there', arr.shape)
        b.append(arr)

#flauta,piano,trumpet,violina, njihov oblak tacaka
for i in b:
    fig=plot_point_cloud(i)
    fig.show()

b_reshaped = []

for i in b:
    k = i.reshape(1, *i.shape)
    # assert isinstance(k, object)
    b_reshaped.append(k)

#pravimo VietorisRips simpleks    i racunamo diagrame perzistencije u diagrami
diagrami = []

VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2], max_edge_length=25)
for i in tqdm(b_reshaped):
    j = VR.fit_transform(i)
   # l=scaler.fit_transform(j)
    diagrami.append(j)


directory = 'vezbe_13'
if not os.path.exists(directory):
    os.makedirs(directory)

for i, diagrams in enumerate(diagrami):
    plotly_slika = plot_diagram(diagrams[0])
    plotly_slika.write_image(f'{directory}/dijagram_{str(i)}.png')


# sigma=0.5,n_bins=60
kernel = HeatKernel()
Heat_diagrams = []

for diagram in tqdm(diagrami):
    heat_diagram = kernel.fit_transform(diagram)[0]
    print('ji', heat_diagram.shape)

    Heat_diagrams.append(heat_diagram)

#for heat in Heat_diagrams:
plt.imshow(Heat_diagrams[1][1],cmap="jet")
plt.show()
plt.close()
# plt.imshow(heat[2],cmap="jet")
# plt.show()
# plt.close()
