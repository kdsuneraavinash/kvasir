# Fast AI Course Notes

## Lesson 1

Every notebook starts with the following three lines; they ensure that any edits to libraries you make are reloaded here automatically, and also that any charts or images displayed are shown in this notebook.

```python
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```

The `fastai` library provides many useful functions that enable us to quickly and easily build neural networks and train our models. `fastai.vision` is the corresponding `fastai` module for vision based applications.

```python
from fastai.vision import *
```

### Loading Dataset

If the dataset is in `.tar.gz` format, we can use `untar_data` function to download and extract data. Otherwise we have to do downloading and extracting using terminal commands.

```python
# if in tar format - this will download to .fastai dir by default
dataset_dir = untar_data(url_file_name)
# if in zip format - will skip ignoring files
!wget -nc {str(url_file_name)} {str(zip_file_name)}
!unzip -n -q {str(zip_file_name)} -d {str(dataset_dir)}
```

`fastai` contains urls of popular datasets. They are in `URLs`.

```python 
URLs.PETS # Pets dataset url link
```

Path objects are easy to use and helps to execute directory commands directly in python easily.

```python
path = urllib.Path('content') # Create path object
data_path = path/'additional'/'data' # Refers to content/additional/data
path.ls() # list files in path
```

We can get image files inside a directly using `get_image_files` function.

```python
fnames = get_image_files(dataset_dir)
```

If image files are in a structure like, `images/cat_001.jpg`, `images/cat_002.jpg`, `images/cat_003.jpg`, `images/dog_001.jpg`, `images/dog_002.jpg` then create dataset using `from_name_re` function.

If image files are in separate folders as `cat`, `dog` use `from_folder`.

```python
# dataset_directory = where dataset is(models and other files are created here)
# ds_tfms = transformers to use (use get_transforms())
# size = width and height of images (standard is 224)
# bs = batch size

# image_files_list = list of image files (preferably taken fromget_image_files)
# pattern = regex pattern to get class name (first group is the class name)
data = ImageDataBunch.from_name_re(dataset_directory, image_files_list, r'\/([a-z])*\.jpg$', ds_tfms=get_transforms(), size=224, bs=32)

# valid_pct = validation set percentage. If image files are in separate `train`, `test` directories, use `train`,.. parameters instead.
data = ImageDataBunch.from_folder(dataset_directory, valid_pct=0.2, ds_tfms=get_transforms(), size=224, bs=32)

# Always normalize images according to some method.
# use imagenet_stats for photos
data.normalize(imagenet_stats)
```

See documentation of a function/class using `doc()` method.

```python
doc(ImageDataBunch.from_name_re)
```

Show sample data from image data bunch.

```python
data.show_batch(rows=3, fig_size=(7, 6))
```

To see identifies class names,

```python
data.classes
# or data.c to see number of classes(in classification)
```

### Training Dataset

To define a learner using a standard model,

```python
# To use metrics,
from fastai.metrics import *

learn = cnn_learner(data, models.resnet34, metrics=accuracy)

# To use several metrics at once,
learn = cnn_learner(data, models.resnet34, metrics=[accuracy, error_rate])

# To show graph of learner stats during training
learn = cnn_learner(data, models.resnet34, metrics=accuracy, callback_fns=ShowGraph)
```

You can view the model architecture using,

```python
learn.model
```

To train the model using one cycle method,

```python 
learn.fit_one_cycle(4) # 4 is number of epochs
```

To save and load the model, (Model will be saves in the directory of dataset inside `models` directory)

```python
learn.save("model-name")

learn.load("model-name")
```

### Evaluating Results

To evaluate a model we need a interpreter.

```python
interp = ClassificationInterpretation.from_learner(learn)
```

We can visualize model stats using

```python
interp.plot_top_losses(9, figsize=(15,11))

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

interp.most_confused(min_val=2)
```

### Fine tuning

To fine tune a model, unfreeze the model and train again. But use `lr_find` to find the best learning rate range.

```python
learn.lr_find()
learn.recorder.plot()
```

Choose the learning rate range from the slope downwards towards a minima and where the slope is max. 

```python
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
```

`1e-6` is start point of slope(highest slope)  and second number is 10 times smaller than the learning rate used in unfrozen stage(as a rule of thumb)

So method generally is,

```python
learn.fit_one_cycle(2, 3e-3) # 3e-3 is the default
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(xxx, 3e-4)) # we have to find xxx by lr_find strongest slope
```

### Other formats

#### Image Data Bunch

[Docs](https://docs.fast.ai/vision.data.html#ImageDataBunch)

| Factory Method                  | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| `ImageDataBunch.from_folder`    | Create from dataset in with `train`,`valid`,`test` sub folders or valid pct |
| `ImageDataBunch.from_csv`       | Create a dataset from a csv file                             |
| `ImageDataBunch.from_df`        | Create a dataset from a dataframe                            |
| `ImageDataBunch.from_name_re`   | Gets the labels from the filenames using a regular expression |
| `ImageDataBunch.from_name_func` | Gets the labels from the filenames using any function        |
| `ImageDataBunch.from_lists`     | Create from list of `fnames` in `path`                       |

#### Transformers

These augments data to help the model to generalize more.

```python
get_transforms(do_flip=True, flip_vert=False, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
```

| Parameter      | Description                                            | Probability  |
| -------------- | ------------------------------------------------------ | ------------ |
| `do_flip`      | whether to flip horizontally                           | `0.5`        |
| `flip_vert`    | whether to flip vertically (requires `do_flip`=`True`) | `0.5`        |
| `max_rotate`   | how much to rotate image                               | `p_affine`   |
| `max_zoom`     | how much to zoom image                                 | `p_affine`   |
| `max_warp`     | how much to warp image                                 | `p_affine`   |
| `max_lighting` | how much to lighten image                              | `p_lighting` |

## Lesson 2

### Collecting a dataset from Google Images

Search in Google images and scroll down until "Show more" button. (Google images will load 700 images initially) Then paste  following command in JavaScript console. This will save image urls in a csv file.

```javascript
urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
```

Then create empty directories for each class, download files and verify files (Since not all urls refer to image files).

```python
classes = ["black", "teddys", "grizzly"]
dest = Path("data/bears")
for class_n in classes:
    class_p = dest/class_n
    csv_p = f"data/{class_n}_urls.csv"
    class_p.mkdir(parents=True, exist_ok=True)
    download_images(csv_p, class_p, max_pics=200)
    verify_images(class_p, delete=True, max_size=500)
```

Now CNN can be trained as usual.

### Cleaning up the dataset

**Currently does not work in collab or GCP**

After an initial training we may need to inspect and remove files which are wrong images. We use `ImageCleaner` widget for that.

```python
from fastai.widgets import *

# create new dataset without splitting (unified dataset)
db = (ImageList.from_folder(path)
                   .split_none()
                   .label_from_folder()
                   .transform(get_transforms(), size=224)
                   .databunch()
     )

# Create cleaner CNN (to find top losses)
learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)
# Load previous trained model (or you can train this from scratch)
learn_cln.load('stage-2');
```

Then load top losses.

```python
# Get top lossed images
ds, idxs = DatasetFormatter().from_toplosses(learn_cln)

# Show image cleaner widget
ImageCleaner(ds, idxs, path)
```

Flag photos for deletion by clicking 'Delete'. Then click 'Next Batch' to delete flagged photos and keep the rest in that row. 

We can also load similar images to delete them.

```python
# Get top lossed images
ds, idxs = DatasetFormatter().from_similars(learn_cln)

# Show image cleaner widget
ImageCleaner(ds, idxs, path, duplicates=True)
```

After that we can create the cleaned `DataBunch` using,

```python
data = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
```

Also if cleaning again, load the unified data bunch using following method, (or previous cleaned stats will be overwritten)

```python
db = (ImageList.from_csv(path, 'cleaned.csv', folder='.')
                   .no_split()
                   .label_from_df()
                   .transform(get_transforms(), size=224)
                   .databunch()
     )
```

### Putting the model in production

First export the production model.

```python
learn.export()
```

This will create a file named `export.pkl` in the directory where we were working that contains everything we need to deploy our model.

In the server, set the default device to `cpu`. (This is the default if the server does not have a `gpu`)

```python
defaults.device = torch.device('cpu')
```

Then load the learner and predict.

```python
# Load the learner (do this once and use??)
learn = load_learner(path)

# Then predict
img = open_image("image.jpg")
pred_class, pred_idx, outputs = learn.predict(img)
print(pred_class)
```

We can use [Starlette](https://www.starlette.io/) to create web apps and [Python Anywhere](https://www.pythonanywhere.com/) to host.

### What could go wrong

#### Learning rate too high

- Validation loss gets a huge number (thousands or maybe millions)
- If this happens, we have to restart from scratch (model will become useless)

#### Learning rate too low

- Error rate or accuracy improves really slowly
- `learn.recorder.plot_losses()` will decrease really slowly
- Training loss > Validation loss
  - This could also mean that number of epochs is low

#### Too few epochs

- Training loss > Validation loss

#### Too many epochs

- This may result in over-fitting
- **It is really hard to over-fit in deep learning**
- Error rate improves(decreases) for a while and starts getting worse again
- Training loss < Validation loss *does not mean that it has over-fit*.

### Theory: Tensors, Linear Regression and SGD

#### Tensors

Tensors are basically regular shaped multi-dimensional arrays. Vectors are rank 1 tensors, matrices are rank 2 tensors.

To practice with tensors and such import `fatsai.basics`.

```python
from fastai.basics import *
```

To create a tensor on ones, use `torch.ones`. `torch.zeros` is also the same. `torch.rand` generates random numbers.

```python
'''
shape = (1,)
[1]
'''
torch.ones(1)

'''
shape = (2, 3)
[1., 1., 1.],
[1., 1., 1.]]
'''
torch.ones(2,3) # shape = (2, 3) =>
```

Each tensor can be treated as a `numpy` array. 

Tensors can be filled with random values using `x.uniform_(min, max)`. Note that underscore at the end refers to in-place functions.

```python
x = torch.ones(100, 2) # 100x2 matrix filled with ones
x[:,0].uniform_(-1, 1) # Fill first column in all rows with random between -1 and 1
```

Tensors can also be defined directly using the array.

```python
a = tensor(2., 3.) # refers to <2, 3> vector(column matrix), here 2 and 3 are not sizes (2x1 matrix)
```

#### Tensor Operations

Tensors can be multiplied and added. `@` is matrix/vector multiplication.

```python
y = x@a + torch.rand(100) # (100x2 mat)*(2x1 mat) + (100x1 mat)
```

Now the problem is that we know `x` and `y` and we need to find `a`.

#### Mean Squared Error Function

This is the error function which is defined as average of squared differences between predicted and real y values.

```python
def mse(y, y_hat):
    return ((y_hat-y)**2).mean()
```

MSE will define how close we are to our target.

What we will do is, we will 'guess' a `a` and then create `y_hat` form `y = x@a`  and then check MSE to check if we are close to the real `a`.

#### Gradient Descent

In order to find the real `a` value, we will use gradient descent.

In order to do that we have to activate gradient recording for the tensor. To create such a tensor use `nn.Parameter`.

```python
a = nn.Parameter(a) # Here `a` is a tensor of shape we want = tensor(-1, 1)
```

Then we can use gradient descent to get to the minima.

```python
def update():
    y_hat = x@a
    loss = mse(y, y_hat)
    print(loss)
    loss.backward()
    with torch.no_grad():
        a.sub_(lr*a.grad)
        a.grad.zero_()
```

Now by updating a several times, we can get to the correct approximation of a.

```python
lr = 1e-1
for i in range(100):
    update()
```

#### Jargon

##### Learning Rate

The **learning rate** or *step size* in [machine learning](https://en.wikipedia.org/wiki/Machine_learning) is a [hyperparameter](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) which determines to what extent newly acquired information overrides old information.

##### Epoch

An **epoch** is a measure of the number of times all of the training vectors are used once to update the weights.

##### Minibatch

When training data is split into small batches, each batch is jargoned as a minibatch. I.e., 1<`size(minibatch)`<`size(trainingdata)`.

[Quora](https://www.quora.com/What-is-a-minibatch-in-a-neural-network)

##### SGD

**Stochastic gradient descent** (often abbreviated **SGD**) is an iterative method for optimizing an objective function with suitable smoothness properties (e.g. differentiable or subdifferentiable). It is called **stochastic** because the method uses randomly selected (or shuffled) samples to evaluate the gradients, hence SGD can be regarded as a [stochastic approximation](https://en.wikipedia.org/wiki/Stochastic_approximation) of [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) optimization.

##### Loss Function

A loss function or cost function is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event. An optimization problem seeks to minimize a loss function. 

