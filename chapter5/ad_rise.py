#%%
import numpy as np
from skimage.transform import resize
from skimage import measure

def _regular_segments(**kwargs):
    s = kwargs["s"]
    input_size = kwargs["img"].shape
    cell_size = np.ceil(np.array(input_size) / s).astype(int)
    segments = np.array(range(s * s)).reshape(s, s)
    up_size = (s + 1) * cell_size
    return segments, up_size

def _slic_segmentation(**kwargs):
    return None

def _segment_image(img, segmentation, **kwargs):
    # Map string arguments to predefined functions
    if isinstance(segmentation, str):
        if segmentation == "regular":
            segmentation_function = _regular_segments
        elif segmentation == "slic":
            segmentation_function = _slic_segmentation
        else:
            raise ValueError("Unsupported segmentation method.")
    elif callable(segmentation):
        segmentation_function = segmentation
    else:
        raise TypeError("Segmentation must be a string or a callable function.")

    return segmentation_function(img=img, **kwargs)

def _sample_segments(segments, N, p, sampling_type="random", segment_probs=None):
    unique_segments = np.unique(segments)
    n_segments = unique_segments.shape[0]
    n_sampled_segments = [
        np.random.choice(
            unique_segments,
            int(p * n_segments),
            False,
            p=segment_probs
        ) 
        for _ in range(N)
    ]
    
    return n_sampled_segments

def _get_number_level_sets(img):
    level_sets = measure.label(img, connectivity=1, background=-1)
    return level_sets.max()

def generate_masks(img: np.array, N: int, s: int=None, p: float=None, segmentation: str="regular", sampling: str="random", over_size: bool=True):

    img_clust = img.copy()
    img_clust = np.squeeze(img_clust)
    if img_clust.shape[0] == 3:
        img_clust = np.moveaxis(img_clust, 0, -1)
    input_size = img_clust.shape
    if len(input_size)==3:
        input_size = input_size[:2]

    segments, up_size = _segment_image(img, segmentation, s=s)
    cell_size = segments.shape
    masker = np.vectorize(lambda x: x in sampled_segments)

    n_sampled_segments = _sample_segments(segments, N, p, sampling_type="random", segment_probs=None)
    # if sampling == "random":
    #     n_sampled_segments = _sampling(segments, N, p, segment_probs=None)
    # elif sampling == "proportional level-sets":
        
    
    masks = np.zeros((N,)+input_size)
    for i, sampled_segments in enumerate(n_sampled_segments):
        mask = masker(segments)
        if over_size is True:
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            masks[i,:,:] = resize(mask.astype('float32'), up_size, order=1, mode='reflect',
                                    anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
        else:
            masks[i,:,:] = resize(mask.astype('float32'), input_size, order=1, mode='reflect',
                                    anti_aliasing=False)
    return masks
        
        
# %%
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
img = np.array(Image.open("pascal_voc/2008_001248.jpg"))
img_gs = np.array(Image.open("pascal_voc/2008_001248.jpg").convert("L"))
masks = generate_masks(img=img_gs, N=2, s=5, p=0.5, segmentation="regular", sampling="random", over_size=True)

masked_img = np.multiply(img, masks[0,:,:].reshape(*img_gs.shape,1)).astype(int)
plt.figure()
plt.imshow(np.squeeze(masked_img))
plt.show()

# %%
from matplotlib import pyplot as plt


# %%
plt.figure()
plt.imshow(np.squeeze(masks[0,:,:]))
plt.show()
# %%
plt.figure()
plt.imshow(img)
plt.show()
# %%
