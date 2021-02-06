using Ailess.Kernel;
using System;
using System.Collections.Generic;

namespace Ailess
{
    class Program
    {
        static void Main(string[] args)
        {
            var task = new ImageClassificationTask();
            task.Run(new Dictionary<string, string>
            {
                ["train_data_dir"] = @"C:\Users\haipi\Pictures\flower_photos"
            });

            Console.WriteLine("Hello World!");
            Console.ReadLine();
        }
    }
}
