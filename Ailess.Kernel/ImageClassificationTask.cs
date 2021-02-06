using Ailess.Task.ImageClassification;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ailess.Kernel
{
    public class ImageClassificationTask : IAilessTask
    {
        public void Run(Dictionary<string, string> args)
        {
            var classifier = new BasicImageClassifier(new ImageClassifierArgs
            {
                TrainDataDir = args["train_data_dir"]
            });

            classifier.Preprocess();
            classifier.BuildModel();
            classifier.Train();
        }
    }
}
