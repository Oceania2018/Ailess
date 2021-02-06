using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Ailess.Task.ImageClassification
{
    public class BasicImageClassifier
    {
        TensorShape img_dim = (180, 180);
        IDatasetV2 train_ds, val_ds;
        Model model;
        ImageClassifierArgs _args;
        int num_classes = -1;

        public BasicImageClassifier(ImageClassifierArgs args)
        {
            _args = args;
            tf.enable_eager_execution();
            num_classes = Directory.GetDirectories(args.TrainDataDir).Length;
        }

        public void BuildModel()
        {
            // var normalization_layer = tf.keras.layers.Rescaling(1.0f / 255);
            var layers = keras.layers;
            model = keras.Sequential(new List<ILayer>
            {
                layers.Rescaling(1.0f / 255, input_shape: (img_dim.dims[0], img_dim.dims[1], 3)),
                layers.Conv2D(16, 3, padding: "same", activation: keras.activations.Relu),
                layers.MaxPooling2D(),
                /*layers.Conv2D(32, 3, padding: "same", activation: "relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding: "same", activation: "relu"),
                layers.MaxPooling2D(),*/
                layers.Flatten(),
                layers.Dense(128, activation: keras.activations.Relu),
                layers.Dense(num_classes)
            });

            model.compile(optimizer: keras.optimizers.Adam(),
                loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                metrics: new[] { "accuracy" });

            model.summary();
        }

        public void Train()
        {
            model.fit(train_ds, validation_data: val_ds, epochs: _args.Epochs);
        }

        public void Preprocess()
        {
            train_ds = keras.preprocessing.image_dataset_from_directory(_args.TrainDataDir,
                validation_split: _args.ValidationSplit,
                subset: "training",
                seed: 123,
                image_size: img_dim,
                batch_size: _args.BatchSize);

            val_ds = keras.preprocessing.image_dataset_from_directory(_args.TrainDataDir,
                validation_split: _args.ValidationSplit,
                subset: "validation",
                seed: 123,
                image_size: img_dim,
                batch_size: _args.BatchSize);

            train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size: -1);
            val_ds = val_ds.cache().prefetch(buffer_size: -1);

            foreach (var (img, label) in train_ds)
            {
                print($"images: {img.TensorShape}");
                print($"labels: {label.numpy()}");
            }
        }
    }
}
