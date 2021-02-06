using System;
using System.Collections.Generic;
using System.Text;

namespace Ailess.Task.ImageClassification
{
    public class ImageClassifierArgs
    {
        public string TrainDataDir { get; set; }
        public int BatchSize { get; set; } = 32;
        public int Epochs { get; set; } = 10;
        public float ValidationSplit { get; set; } = 0.2f;
    }
}
