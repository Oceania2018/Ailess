using System;
using System.Collections.Generic;
using System.Text;

namespace Ailess.Kernel
{
    public interface IAilessTask
    {
        void Run(Dictionary<string, string> args);
    }
}
